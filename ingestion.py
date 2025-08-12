
import asyncio
import json
import logging
import uuid
from typing import List

import boto3
import pandas as pd
from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from langchain_postgres import PGEngine
from sqlalchemy import Column, String, DateTime
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import UUID, insert

# Set up logging for production-grade monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
bedrock_runtime = boto3.client("bedrock-runtime")

# Constants
POSTGRES_USER = "langchain"
POSTGRES_PASSWORD = "langchain"
POSTGRES_HOST = "your-rds-endpoint.region.rds.amazonaws.com"
POSTGRES_PORT = "5432"
POSTGRES_DB = "langchain"
CONNECTION_STRING = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
SCHEMA_NAME = "test"
TABLE_NAME = "vectorstore"
VECTOR_SIZE = 1536  # Dimension for Amazon Titan Embed Text v1
BATCH_SIZE = 100   # Batch size for ingestion; adjust based on Bedrock limits (max 25 per request for Titan Embed)
DYNAMODB_TABLE = "FileMetadata"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"

# Initialize embedding model globally
embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_ID,
    client=bedrock_runtime
)

async def create_schema_if_not_exists(engine: PGEngine):
    """Create the schema if it doesn't exist."""
    async with engine.engine.connect() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))
        await conn.commit()
    logger.info(f"Schema '{SCHEMA_NAME}' ensured.")

async def initialize_table(engine: PGEngine):
    """Initialize the vector store table with custom metadata columns."""
    custom_columns = [
        Column("docid", String, unique=True),  # Unique document ID
        Column("creator", String),            # Creator of the document
        Column("creationdate", DateTime),     # Creation date
    ]
    await engine.ainit_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=VECTOR_SIZE,
        schema_name=SCHEMA_NAME,
        id_column=Column("langchain_id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        metadata_columns=custom_columns
    )
    logger.info(f"Table '{TABLE_NAME}' initialized in schema '{SCHEMA_NAME}'.")

async def update_dynamodb_status(file_key: str, status: str, error_message: str = None):
    """Update file status in DynamoDB."""
    table = dynamodb.Table(DYNAMODB_TABLE)
    update_expression = "SET #status = :status"
    expression_values = {":status": status}
    expression_names = {"#status": "status"}
    
    if error_message:
        update_expression += ", error_message = :error"
        expression_values[":error"] = error_message

    try:
        table.update_item(
            Key={"file_key": file_key},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values,
            ExpressionAttributeNames=expression_names
        )
        logger.info(f"Updated DynamoDB status for {file_key} to {status}")
    except Exception as e:
        logger.error(f"Failed to update DynamoDB for {file_key}: {e}")
        raise

async def process_excel_file(bucket: str, file_key: str) -> List[Document]:
    """Read Excel file from S3 and convert rows to Documents."""
    try:
        # Download file from S3
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        df = pd.read_excel(response["Body"])
        
        documents = []
        docids = []
        for _, row in df.iterrows():
            docid = str(row.get("docid", uuid.uuid4()))  # Use provided docid or generate one
            metadata = {
                "docid": docid,
                "creator": str(row.get("creator", "unknown")),
                "creationdate": pd.to_datetime(row.get("creationdate", pd.Timestamp.now())).isoformat(),
            }
            # Combine Description, Title, Notes for embedding
            content_fields = [
                str(row.get("Description", "")),
                str(row.get("Title", "")),
                str(row.get("Notes", ""))
            ]
            content = " ".join([field for field in content_fields if field])
            
            documents.append(Document(
                id=str(uuid.uuid4()),  # langchain_id for vector store
                page_content=content,
                metadata=metadata
            ))
            docids.append(docid)
        
        return documents, docids
    except Exception as e:
        logger.error(f"Failed to process Excel file {file_key}: {e}")
        raise

async def transactional_upsert(engine: PGEngine, documents: List[Document], docids: List[str]):
    """
    Perform delete and insert in a single transaction to ensure atomicity.
    If insert fails, rollback the delete to prevent data loss.
    """
    # Generate embeddings first (outside transaction to minimize lock time)
    contents = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    langchain_ids = [str(uuid.uuid4()) for _ in documents]  # Generate new langchain_ids
    embedded_vectors = []
    for i in range(0, len(contents), 25):  # Bedrock Titan Embed limit is 25 per request
        batch_contents = contents[i:i + 25]
        embedded_vectors.extend(embeddings.embed_documents(batch_contents))
    logger.info(f"Generated embeddings for {len(documents)} documents.")

    # Prepare insert data as list of dicts
    insert_data = []
    for lang_id, content, embedding, metadata in zip(langchain_ids, contents, embedded_vectors, metadatas):
        insert_data.append({
            "langchain_id": lang_id,
            "content": content,
            "embedding": embedding,  # List of floats for VECTOR type
            "docid": metadata["docid"],
            "creator": metadata["creator"],
            "creationdate": metadata["creationdate"]
        })

    async with engine.begin() as conn:  # Begins a transaction
        # Delete existing docs
        await conn.execute(
            text(f"DELETE FROM {SCHEMA_NAME}.{TABLE_NAME} WHERE docid = ANY(:docids)"),
            {"docids": docids}
        )

        # Manual batch insert
        table = engine.metadata.tables[f"{SCHEMA_NAME}.{TABLE_NAME}"]
        insert_stmt = insert(table)
        for i in range(0, len(insert_data), BATCH_SIZE):
            batch = insert_data[i:i + BATCH_SIZE]
            await conn.execute(insert_stmt, batch)
            logger.info(f"Inserted batch {i // BATCH_SIZE + 1} of {len(insert_data) // BATCH_SIZE + 1}")

        # Commit happens automatically on successful exit

async def ingest_documents(bucket: str, file_key: str) -> None:
    """Main ingestion logic with transactional upsert."""
    engine = PGEngine.from_connection_string(
        url=CONNECTION_STRING,
        pool_pre_ping=True,
        pool_size=20,
        pool_timeout=30,
        max_overflow=10
    )

    try:
        # Update DynamoDB status to Processing
        await update_dynamodb_status(file_key, "Processing")
        
        # Create schema and table
        await create_schema_if_not_exists(engine)
        await initialize_table(engine)
        
        # Process Excel file
        documents, docids = await process_excel_file(bucket, file_key)
        
        # Perform transactional upsert
        if documents:
            await transactional_upsert(engine, documents, docids)
        
        # Update DynamoDB status to Completed
        await update_dynamodb_status(file_key, "Completed")
        logger.info(f"Ingestion completed for {file_key}")
    
    except Exception as e:
        # Log error and update DynamoDB with error message
        error_message = str(e)
        await update_dynamodb_status(file_key, "Failed", error_message)
        raise
    finally:
        await engine.engine.dispose()

def lambda_handler(event, context):
    """AWS Lambda handler for S3 event."""
    try:
        # Extract bucket and file key from S3 event
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            file_key = record["s3"]["object"]["key"]
            
            # Run async ingestion
            asyncio.run(ingest_documents(bucket, file_key))
        
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Ingestion completed"})
        }
    except Exception as e:
        logger.error(f"Lambda failed: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
