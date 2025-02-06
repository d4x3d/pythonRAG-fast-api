-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documents table to track processed documents
create table if not exists documents (
    id bigserial primary key,
    file_hash text unique not null,
    filename text not null,
    processed_at timestamp with time zone default current_timestamp,
    total_chunks integer not null,
    total_tokens integer not null
);

-- Create the document_sections table for vector storage
create table if not exists document_sections (
    id bigserial primary key,
    content text,
    metadata jsonb,
    embedding vector(384)
);

-- Create indexes
create index if not exists idx_document_sections_embedding on document_sections 
using hnsw (embedding vector_cosine_ops);

create index if not exists idx_document_sections_metadata on document_sections 
using gin (metadata);

create index if not exists idx_documents_file_hash on documents(file_hash);

-- Create a function to check if a document exists
create or replace function check_document_exists(hash text)
returns boolean as $$
begin
    return exists(
        select 1 
        from documents 
        where file_hash = hash
    );
end;
$$ language plpgsql;

-- Create chat sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id bigserial primary key,
    session_id text unique not null,
    created_at timestamp with time zone default current_timestamp
);

-- Create table to track which documents are part of each chat session
CREATE TABLE IF NOT EXISTS chat_session_documents (
    id bigserial primary key,
    session_id text references chat_sessions(session_id),
    file_hash text references documents(file_hash),
    added_at timestamp with time zone default current_timestamp,
    UNIQUE(session_id, file_hash)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_chat_session_documents_session 
ON chat_session_documents(session_id);

-- Create a function to get document sections for a specific chat session
CREATE OR REPLACE FUNCTION get_session_document_sections(p_session_id text)
RETURNS TABLE (
    id bigint,
    content text,
    metadata jsonb,
    embedding vector(384)
) AS $$
BEGIN
    RETURN QUERY
    SELECT ds.*
    FROM document_sections ds
    JOIN chat_session_documents csd 
        ON ds.metadata->>'file_hash' = csd.file_hash
    WHERE csd.session_id = p_session_id;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
grant all privileges on all tables in schema public to postgres;
grant all privileges on all sequences in schema public to postgres;
