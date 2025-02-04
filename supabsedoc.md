LangChain
LangChain is a popular framework for working with AI, Vectors, and embeddings. LangChain supports using Supabase as a vector store, using the pgvector extension.

Initializing your database#
Prepare you database with the relevant tables:


Dashboard

SQL
-- Enable the pgvector extension to work with embedding vectors
create extension vector;

-- Create a table to store your documents
create table documents (
  id bigserial primary key,
  content text, -- corresponds to Document.pageContent
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(1536) -- 1536 works for OpenAI embeddings, change if needed
);

-- Create a function to search for documents
create function match_documents (
  query_embedding vector(1536),
  match_count int default null,
  filter jsonb DEFAULT '{}'
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

Usage#
You can now search your documents using any Node.js application. This is intended to be run on a secure server route.

import { SupabaseVectorStore } from 'langchain/vectorstores/supabase'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { createClient } from '@supabase/supabase-js'

const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY
if (!supabaseKey) throw new Error(`Expected SUPABASE_SERVICE_ROLE_KEY`)

const url = process.env.SUPABASE_URL
if (!url) throw new Error(`Expected env var SUPABASE_URL`)

export const run = async () => {
  const client = createClient(url, supabaseKey)

  const vectorStore = await SupabaseVectorStore.fromTexts(
    ['Hello world', 'Bye bye', "What's this?"],
    [{ id: 2 }, { id: 1 }, { id: 3 }],
    new OpenAIEmbeddings(),
    {
      client,
      tableName: 'documents',
      queryName: 'match_documents',
    }
  )

  const resultOne = await vectorStore.similaritySearch('Hello world', 1)

  console.log(resultOne)
}

Simple metadata filtering#
Given the above match_documents Postgres function, you can also pass a filter parameter to only return documents with a specific metadata field value. This filter parameter is a JSON object, and the match_documents function will use the Postgres JSONB Containment operator @> to filter documents by the metadata field values you specify. See details on the Postgres JSONB Containment operator for more information.

import { SupabaseVectorStore } from 'langchain/vectorstores/supabase'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { createClient } from '@supabase/supabase-js'

// First, follow set-up instructions above

const privateKey = process.env.SUPABASE_SERVICE_ROLE_KEY
if (!privateKey) throw new Error(`Expected env var SUPABASE_SERVICE_ROLE_KEY`)

const url = process.env.SUPABASE_URL
if (!url) throw new Error(`Expected env var SUPABASE_URL`)

export const run = async () => {
  const client = createClient(url, privateKey)

  const vectorStore = await SupabaseVectorStore.fromTexts(
    ['Hello world', 'Hello world', 'Hello world'],
    [{ user_id: 2 }, { user_id: 1 }, { user_id: 3 }],
    new OpenAIEmbeddings(),
    {
      client,
      tableName: 'documents',
      queryName: 'match_documents',
    }
  )

  const result = await vectorStore.similaritySearch('Hello world', 1, {
    user_id: 3,
  })

  console.log(result)
}

Advanced metadata filtering#
You can also use query builder-style filtering (similar to how the Supabase JavaScript library works) instead of passing an object. Note that since the filter properties will be in the metadata column, you need to use arrow operators (-> for integer or ->> for text) as defined in Postgrest API documentation and specify the data type of the property (e.g. the column should look something like metadata->some_int_value::int).

import { SupabaseFilterRPCCall, SupabaseVectorStore } from 'langchain/vectorstores/supabase'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { createClient } from '@supabase/supabase-js'

// First, follow set-up instructions above

const privateKey = process.env.SUPABASE_SERVICE_ROLE_KEY
if (!privateKey) throw new Error(`Expected env var SUPABASE_SERVICE_ROLE_KEY`)

const url = process.env.SUPABASE_URL
if (!url) throw new Error(`Expected env var SUPABASE_URL`)

export const run = async () => {
  const client = createClient(url, privateKey)

  const embeddings = new OpenAIEmbeddings()

  const store = new SupabaseVectorStore(embeddings, {
    client,
    tableName: 'documents',
  })

  const docs = [
    {
      pageContent:
        'This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theoretical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.',
      metadata: { b: 1, c: 10, stuff: 'right' },
    },
    {
      pageContent:
        'This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.',
      metadata: { b: 2, c: 9, stuff: 'right' },
    },
    { pageContent: 'hello', metadata: { b: 1, c: 9, stuff: 'right' } },
    { pageContent: 'hello', metadata: { b: 1, c: 9, stuff: 'wrong' } },
    { pageContent: 'hi', metadata: { b: 2, c: 8, stuff: 'right' } },
    { pageContent: 'bye', metadata: { b: 3, c: 7, stuff: 'right' } },
    { pageContent: "what's this", metadata: { b: 4, c: 6, stuff: 'right' } },
  ]

  await store.addDocuments(docs)

  const funcFilterA: SupabaseFilterRPCCall = (rpc) =>
    rpc
      .filter('metadata->b::int', 'lt', 3)
      .filter('metadata->c::int', 'gt', 7)
      .textSearch('content', `'multidimensional' & 'spaces'`, {
        config: 'english',
      })

  const resultA = await store.similaritySearch('quantum', 4, funcFilterA)

  const funcFilterB: SupabaseFilterRPCCall = (rpc) =>
    rpc
      .filter('metadata->b::int', 'lt', 3)
      .filter('metadata->c::int', 'gt', 7)
      .filter('metadata->>stuff', 'eq', 'right')

  const resultB = await store.similaritySearch('hello', 2, funcFilterB)

  console.log(resultA, resultB)
}

Vector columns
Supabase offers a number of different ways to store and query vectors within Postgres. The SQL included in this guide is applicable for clients in all programming languages. If you are a Python user see your Python client options after reading the Learn section.

Vectors in Supabase are enabled via pgvector, a PostgreSQL extension for storing and querying vectors in Postgres. It can be used to store embeddings.

Usage#
Enable the extension#

Dashboard

SQL
 -- Example: enable the "vector" extension.
create extension vector
with
  schema extensions;

-- Example: disable the "vector" extension
drop
  extension if exists vector;

Even though the SQL code is create extension, this is the equivalent of "enabling the extension".
To disable an extension, call drop extension.

Create a table to store vectors#
After enabling the vector extension, you will get access to a new data type called vector. The size of the vector (indicated in parenthesis) represents the number of dimensions stored in that vector.

create table documents (
  id serial primary key,
  title text not null,
  body text not null,
  embedding vector(384)
);

In the above SQL snippet, we create a documents table with a column called embedding (note this is just a regular Postgres column - you can name it whatever you like). We give the embedding column a vector data type with 384 dimensions. Change this to the number of dimensions produced by your embedding model. For example, if you are generating embeddings using the open source gte-small model, you would set this number to 384 since that model produces 384 dimensions.

In general, embeddings with fewer dimensions perform best. See our analysis on fewer dimensions in pgvector.

Storing a vector / embedding#
In this example we'll generate a vector using Transformers.js, then store it in the database using the Supabase JavaScript client.

import { pipeline } from '@xenova/transformers'
const generateEmbedding = await pipeline('feature-extraction', 'Supabase/gte-small')

const title = 'First post!'
const body = 'Hello world!'

// Generate a vector using Transformers.js
const output = await generateEmbedding(body, {
  pooling: 'mean',
  normalize: true,
})

// Extract the embedding output
const embedding = Array.from(output.data)

// Store the vector in Postgres
const { data, error } = await supabase.from('documents').insert({
  title,
  body,
  embedding,
})

This example uses the JavaScript Supabase client, but you can modify it to work with any supported language library.

Querying a vector / embedding#
Similarity search is the most common use case for vectors. pgvector support 3 new operators for computing distance:

Operator	Description
<->	Euclidean distance
<#>	negative inner product
<=>	cosine distance
Choosing the right operator depends on your needs. Dot product tends to be the fastest if your vectors are normalized. For more information on how embeddings work and how they relate to each other, see What are Embeddings?.

Supabase client libraries like supabase-js connect to your Postgres instance via PostgREST. PostgREST does not currently support pgvector similarity operators, so we'll need to wrap our query in a Postgres function and call it via the rpc() method:

create or replace function match_documents (
  query_embedding vector(384),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  title text,
  body text,
  similarity float
)
language sql stable
as $$
  select
    documents.id,
    documents.title,
    documents.body,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by (documents.embedding <=> query_embedding) asc
  limit match_count;
$$;

This function takes a query_embedding argument and compares it to all other embeddings in the documents table. Each comparison returns a similarity score. If the similarity is greater than the match_threshold argument, it is returned. The number of rows returned is limited by the match_count argument.

Feel free to modify this method to fit the needs of your application. The match_threshold ensures that only documents that have a minimum similarity to the query_embedding are returned. Without this, you may end up returning documents that subjectively don't match. This value will vary for each application - you will need to perform your own testing to determine the threshold that makes sense for your app.

If you index your vector column, ensure that the order by sorts by the distance function directly (rather than sorting by the calculated similarity column, which may lead to the index being ignored and poor performance).

To execute the function from your client library, call rpc() with the name of your Postgres function:

const { data: documents } = await supabaseClient.rpc('match_documents', {
  query_embedding: embedding, // Pass the embedding you want to compare
  match_threshold: 0.78, // Choose an appropriate threshold for your data
  match_count: 10, // Choose the number of matches
})

In this example embedding would be another embedding you wish to compare against your table of pre-generated embedding documents. For example if you were building a search engine, every time the user submits their query you would first generate an embedding on the search query itself, then pass it into the above rpc() function to match.

Be sure to use embeddings produced from the same embedding model when calculating distance. Comparing embeddings from two different models will produce no meaningful result

HNSW is an algorithm for approximate nearest neighbor search. It is a frequently used index type that can improve performance when querying highly-dimensional vectors, like those representing embeddings.

Usage#
The way you create an HNSW index depends on the distance operator you are using. pgvector includes 3 distance operators:

Operator	Description	Operator class
<->	Euclidean distance	vector_l2_ops
<#>	negative inner product	vector_ip_ops
<=>	cosine distance	vector_cosine_ops
Use the following SQL commands to create an HNSW index for the operator(s) used in your queries.

Euclidean L2 distance (vector_l2_ops)#
create index on items using hnsw (column_name vector_l2_ops);

Inner product (vector_ip_ops)#
create index on items using hnsw (column_name vector_ip_ops);

Cosine distance (vector_cosine_ops)#
create index on items using hnsw (column_name vector_cosine_ops);

Currently vectors with up to 2,000 dimensions can be indexed.

How does HNSW work?#
HNSW uses proximity graphs (graphs connecting nodes based on distance between them) to approximate nearest-neighbor search. To understand HNSW, we can break it down into 2 parts:

Hierarchical (H): The algorithm operates over multiple layers
Navigable Small World (NSW): Each vector is a node within a graph and is connected to several other nodes
Hierarchical#
The hierarchical aspect of HNSW builds off of the idea of skip lists.

Skip lists are multi-layer linked lists. The bottom layer is a regular linked list connecting an ordered sequence of elements. Each new layer above removes some elements from the underlying layer (based on a fixed probability), producing a sparser subsequence that “skips” over elements.

visual of an example skip list
When searching for an element, the algorithm begins at the top layer and traverses its linked list horizontally. If the target element is found, the algorithm stops and returns it. Otherwise if the next element in the list is greater than the target (or NULL), the algorithm drops down to the next layer below. Since each layer below is less sparse than the layer above (with the bottom layer connecting all elements), the target will eventually be found. Skip lists offer O(log n) average complexity for both search and insertion/deletion.

Navigable Small World#
A navigable small world (NSW) is a special type of proximity graph that also includes long-range connections between nodes. These long-range connections support the “small world” property of the graph, meaning almost every node can be reached from any other node within a few hops. Without these additional long-range connections, many hops would be required to reach a far-away node.

visual of an example navigable small world graph

The “navigable” part of NSW specifically refers to the ability to logarithmically scale the greedy search algorithm on the graph, an algorithm that attempts to make only the locally optimal choice at each hop. Without this property, the graph may still be considered a small world with short paths between far-away nodes, but the greedy algorithm tends to miss them. Greedy search is ideal for NSW because it is quick to navigate and has low computational costs.

Hierarchical + Navigable Small World#
HNSW combines these two concepts. From the hierarchical perspective, the bottom layer consists of a NSW made up of short links between nodes. Each layer above “skips” elements and creates longer links between nodes further away from each other.

Just like skip lists, search starts at the top layer and works its way down until it finds the target element. However, instead of comparing a scalar value at each layer to determine whether or not to descend to the layer below, a multi-dimensional distance measure (such as Euclidean distance) is used.

When should you create HNSW indexes?#
HNSW should be your default choice when creating a vector index. Add the index when you don't need 100% accuracy and are willing to trade a small amount of accuracy for a lot of throughput.

Unlike IVFFlat indexes, you are safe to build an HNSW index immediately after the table is created. HNSW indexes are based on graphs which inherently are not affected by the same limitations as IVFFlat. As new data is added to the table, the index will be filled automatically and the index structure will remain optimal.

pgvector
Open-source vector similarity search for Postgres

Store your vectors with the rest of your data. Supports:

exact and approximate nearest neighbor search
single-precision, half-precision, binary, and sparse vectors
L2 distance, inner product, cosine distance, L1 distance, Hamming distance, and Jaccard distance
any language with a Postgres client
Plus ACID compliance, point-in-time recovery, JOINs, and all of the other great features of Postgres

Build Status

Installation
Linux and Mac
Compile and install the extension (supports Postgres 13+)

cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install # may need sudo
See the installation notes if you run into issues

You can also install it with Docker, Homebrew, PGXN, APT, Yum, pkg, or conda-forge, and it comes preinstalled with Postgres.app and many hosted providers. There are also instructions for GitHub Actions.

Windows
Ensure C++ support in Visual Studio is installed, and run:

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
Note: The exact path will vary depending on your Visual Studio version and edition

Then use nmake to build:

set "PGROOT=C:\Program Files\PostgreSQL\16"
cd %TEMP%
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
Note: Postgres 17 is not supported with MSVC yet due to an upstream issue

See the installation notes if you run into issues

You can also install it with Docker or conda-forge.

Getting Started
Enable the extension (do this once in each database where you want to use it)

CREATE EXTENSION vector;
Create a vector column with 3 dimensions

CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
Insert vectors

INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
Get the nearest neighbors by L2 distance

SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
Also supports inner product (<#>), cosine distance (<=>), and L1 distance (<+>, added in 0.7.0)

Note: <#> returns the negative inner product since Postgres only supports ASC order index scans on operators

Storing
Create a new table with a vector column

CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
Or add a vector column to an existing table

ALTER TABLE items ADD COLUMN embedding vector(3);
Also supports half-precision, binary, and sparse vectors

Insert vectors

INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
Or load vectors in bulk using COPY (example)

COPY items (embedding) FROM STDIN WITH (FORMAT BINARY);
Upsert vectors

INSERT INTO items (id, embedding) VALUES (1, '[1,2,3]'), (2, '[4,5,6]')
    ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding;
Update vectors

UPDATE items SET embedding = '[1,2,3]' WHERE id = 1;
Delete vectors

DELETE FROM items WHERE id = 1;
Querying
Get the nearest neighbors to a vector

SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
Supported distance functions are:

<-> - L2 distance
<#> - (negative) inner product
<=> - cosine distance
<+> - L1 distance (added in 0.7.0)
<~> - Hamming distance (binary vectors, added in 0.7.0)
<%> - Jaccard distance (binary vectors, added in 0.7.0)
Get the nearest neighbors to a row

SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5;
Get rows within a certain distance

SELECT * FROM items WHERE embedding <-> '[3,1,2]' < 5;
Note: Combine with ORDER BY and LIMIT to use an index

Distances
Get the distance

SELECT embedding <-> '[3,1,2]' AS distance FROM items;
For inner product, multiply by -1 (since <#> returns the negative inner product)

SELECT (embedding <#> '[3,1,2]') * -1 AS inner_product FROM items;
For cosine similarity, use 1 - cosine distance

SELECT 1 - (embedding <=> '[3,1,2]') AS cosine_similarity FROM items;
Aggregates
Average vectors

SELECT AVG(embedding) FROM items;
Average groups of vectors

SELECT category_id, AVG(embedding) FROM items GROUP BY category_id;
Indexing
By default, pgvector performs exact nearest neighbor search, which provides perfect recall.

You can add an index to use approximate nearest neighbor search, which trades some recall for speed. Unlike typical indexes, you will see different results for queries after adding an approximate index.

Supported index types are:

HNSW
IVFFlat
HNSW
An HNSW index creates a multilayer graph. It has better query performance than IVFFlat (in terms of speed-recall tradeoff), but has slower build times and uses more memory. Also, an index can be created without any data in the table since there isn’t a training step like IVFFlat.

Add an index for each distance function you want to use.

L2 distance

CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
Note: Use halfvec_l2_ops for halfvec and sparsevec_l2_ops for sparsevec (and similar with the other distance functions)

Inner product

CREATE INDEX ON items USING hnsw (embedding vector_ip_ops);
Cosine distance

CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);
L1 distance - added in 0.7.0

CREATE INDEX ON items USING hnsw (embedding vector_l1_ops);
Hamming distance - added in 0.7.0

CREATE INDEX ON items USING hnsw (embedding bit_hamming_ops);
Jaccard distance - added in 0.7.0

CREATE INDEX ON items USING hnsw (embedding bit_jaccard_ops);
Supported types are:

vector - up to 2,000 dimensions
halfvec - up to 4,000 dimensions (added in 0.7.0)
bit - up to 64,000 dimensions (added in 0.7.0)
sparsevec - up to 1,000 non-zero elements (added in 0.7.0)
Index Options
Specify HNSW parameters

m - the max number of connections per layer (16 by default)
ef_construction - the size of the dynamic candidate list for constructing the graph (64 by default)
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);
A higher value of ef_construction provides better recall at the cost of index build time / insert speed.

Query Options
Specify the size of the dynamic candidate list for search (40 by default)

SET hnsw.ef_search = 100;
A higher value provides better recall at the cost of speed.

Use SET LOCAL inside a transaction to set it for a single query

BEGIN;
SET LOCAL hnsw.ef_search = 100;
SELECT ...
COMMIT;
Index Build Time
Indexes build significantly faster when the graph fits into maintenance_work_mem

SET maintenance_work_mem = '8GB';
A notice is shown when the graph no longer fits

NOTICE:  hnsw graph no longer fits into maintenance_work_mem after 100000 tuples
DETAIL:  Building will take significantly more time.
HINT:  Increase maintenance_work_mem to speed up builds.
Note: Do not set maintenance_work_mem so high that it exhausts the memory on the server

Like other index types, it’s faster to create an index after loading your initial data

Starting with 0.6.0, you can also speed up index creation by increasing the number of parallel workers (2 by default)

SET max_parallel_maintenance_workers = 7; -- plus leader
For a large number of workers, you may also need to increase max_parallel_workers (8 by default)

Indexing Progress
Check indexing progress

SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS "%" FROM pg_stat_progress_create_index;
The phases for HNSW are:

initializing
loading tuples
IVFFlat
An IVFFlat index divides vectors into lists, and then searches a subset of those lists that are closest to the query vector. It has faster build times and uses less memory than HNSW, but has lower query performance (in terms of speed-recall tradeoff).

Three keys to achieving good recall are:

Create the index after the table has some data
Choose an appropriate number of lists - a good place to start is rows / 1000 for up to 1M rows and sqrt(rows) for over 1M rows
When querying, specify an appropriate number of probes (higher is better for recall, lower is better for speed) - a good place to start is sqrt(lists)
Add an index for each distance function you want to use.

L2 distance

CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
Note: Use halfvec_l2_ops for halfvec (and similar with the other distance functions)

Inner product

CREATE INDEX ON items USING ivfflat (embedding vector_ip_ops) WITH (lists = 100);
Cosine distance

CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
Hamming distance - added in 0.7.0

CREATE INDEX ON items USING ivfflat (embedding bit_hamming_ops) WITH (lists = 100);
Supported types are:

vector - up to 2,000 dimensions
halfvec - up to 4,000 dimensions (added in 0.7.0)
bit - up to 64,000 dimensions (added in 0.7.0)
Query Options
Specify the number of probes (1 by default)

SET ivfflat.probes = 10;
A higher value provides better recall at the cost of speed, and it can be set to the number of lists for exact nearest neighbor search (at which point the planner won’t use the index)

Use SET LOCAL inside a transaction to set it for a single query

BEGIN;
SET LOCAL ivfflat.probes = 10;
SELECT ...
COMMIT;
Index Build Time
Speed up index creation on large tables by increasing the number of parallel workers (2 by default)

SET max_parallel_maintenance_workers = 7; -- plus leader
For a large number of workers, you may also need to increase max_parallel_workers (8 by default)

Indexing Progress
Check indexing progress

SELECT phase, round(100.0 * tuples_done / nullif(tuples_total, 0), 1) AS "%" FROM pg_stat_progress_create_index;
The phases for IVFFlat are:

initializing
performing k-means
assigning tuples
loading tuples
Note: % is only populated during the loading tuples phase

Filtering
There are a few ways to index nearest neighbor queries with a WHERE clause.

SELECT * FROM items WHERE category_id = 123 ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
A good place to start is creating an index on the filter column. This can provide fast, exact nearest neighbor search in many cases. Postgres has a number of index types for this: B-tree (default), hash, GiST, SP-GiST, GIN, and BRIN.

CREATE INDEX ON items (category_id);
For multiple columns, consider a multicolumn index.

CREATE INDEX ON items (location_id, category_id);
Exact indexes work well for conditions that match a low percentage of rows. Otherwise, approximate indexes can work better.

CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
With approximate indexes, filtering is applied after the index is scanned. If a condition matches 10% of rows, with HNSW and the default hnsw.ef_search of 40, only 4 rows will match on average. For more rows, increase hnsw.ef_search.

SET hnsw.ef_search = 200;
Starting with 0.8.0, you can enable iterative index scans, which will automatically scan more of the index when needed.

SET hnsw.iterative_scan = strict_order;
If filtering by only a few distinct values, consider partial indexing.

CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WHERE (category_id = 123);
If filtering by many different values, consider partitioning.

CREATE TABLE items (embedding vector(3), category_id int) PARTITION BY LIST(category_id);
Iterative Index Scans
Added in 0.8.0

With approximate indexes, queries with filtering can return less results since filtering is applied after the index is scanned. Starting with 0.8.0, you can enable iterative index scans, which will automatically scan more of the index until enough results are found (or it reaches hnsw.max_scan_tuples or ivfflat.max_probes).

Iterative scans can use strict or relaxed ordering.

Strict ensures results are in the exact order by distance

SET hnsw.iterative_scan = strict_order;
Relaxed allows results to be slightly out of order by distance, but provides better recall

SET hnsw.iterative_scan = relaxed_order;
# or
SET ivfflat.iterative_scan = relaxed_order;
With relaxed ordering, you can use a materialized CTE to get strict ordering

WITH relaxed_results AS MATERIALIZED (
    SELECT id, embedding <-> '[1,2,3]' AS distance FROM items WHERE category_id = 123 ORDER BY distance LIMIT 5
) SELECT * FROM relaxed_results ORDER BY distance;
For queries that filter by distance, use a materialized CTE and place the distance filter outside of it for best performance (due to the current behavior of the Postgres executor)

WITH nearest_results AS MATERIALIZED (
    SELECT id, embedding <-> '[1,2,3]' AS distance FROM items ORDER BY distance LIMIT 5
) SELECT * FROM nearest_results WHERE distance < 5 ORDER BY distance;
Note: Place any other filters inside the CTE

Iterative Scan Options
Since scanning a large portion of an approximate index is expensive, there are options to control when a scan ends.

HNSW
Specify the max number of tuples to visit (20,000 by default)

SET hnsw.max_scan_tuples = 20000;
Note: This is approximate and does not affect the initial scan

Specify the max amount of memory to use, as a multiple of work_mem (1 by default)

SET hnsw.scan_mem_multiplier = 2;
Note: Try increasing this if increasing hnsw.max_scan_tuples does not improve recall

IVFFlat
Specify the max number of probes

SET ivfflat.max_probes = 100;
Note: If this is lower than ivfflat.probes, ivfflat.probes will be used

Half-Precision Vectors
Added in 0.7.0

Use the halfvec type to store half-precision vectors

CREATE TABLE items (id bigserial PRIMARY KEY, embedding halfvec(3));
Half-Precision Indexing
Added in 0.7.0

Index vectors at half precision for smaller indexes

CREATE INDEX ON items USING hnsw ((embedding::halfvec(3)) halfvec_l2_ops);
Get the nearest neighbors

SELECT * FROM items ORDER BY embedding::halfvec(3) <-> '[1,2,3]' LIMIT 5;
Binary Vectors
Use the bit type to store binary vectors (example)

CREATE TABLE items (id bigserial PRIMARY KEY, embedding bit(3));
INSERT INTO items (embedding) VALUES ('000'), ('111');
Get the nearest neighbors by Hamming distance (added in 0.7.0)

SELECT * FROM items ORDER BY embedding <~> '101' LIMIT 5;
Or (before 0.7.0)

SELECT * FROM items ORDER BY bit_count(embedding # '101') LIMIT 5;
Also supports Jaccard distance (<%>)

Binary Quantization
Added in 0.7.0

Use expression indexing for binary quantization

CREATE INDEX ON items USING hnsw ((binary_quantize(embedding)::bit(3)) bit_hamming_ops);
Get the nearest neighbors by Hamming distance

SELECT * FROM items ORDER BY binary_quantize(embedding)::bit(3) <~> binary_quantize('[1,-2,3]') LIMIT 5;
Re-rank by the original vectors for better recall

SELECT * FROM (
    SELECT * FROM items ORDER BY binary_quantize(embedding)::bit(3) <~> binary_quantize('[1,-2,3]') LIMIT 20
) ORDER BY embedding <=> '[1,-2,3]' LIMIT 5;
Sparse Vectors
Added in 0.7.0

Use the sparsevec type to store sparse vectors

CREATE TABLE items (id bigserial PRIMARY KEY, embedding sparsevec(5));
Insert vectors

INSERT INTO items (embedding) VALUES ('{1:1,3:2,5:3}/5'), ('{1:4,3:5,5:6}/5');
The format is {index1:value1,index2:value2}/dimensions and indices start at 1 like SQL arrays

Get the nearest neighbors by L2 distance

SELECT * FROM items ORDER BY embedding <-> '{1:3,3:1,5:2}/5' LIMIT 5;
Hybrid Search
Use together with Postgres full-text search for hybrid search.

SELECT id, content FROM items, plainto_tsquery('hello search') query
    WHERE textsearch @@ query ORDER BY ts_rank_cd(textsearch, query) DESC LIMIT 5;
You can use Reciprocal Rank Fusion or a cross-encoder to combine results.

Indexing Subvectors
Added in 0.7.0

Use expression indexing to index subvectors

CREATE INDEX ON items USING hnsw ((subvector(embedding, 1, 3)::vector(3)) vector_cosine_ops);
Get the nearest neighbors by cosine distance

SELECT * FROM items ORDER BY subvector(embedding, 1, 3)::vector(3) <=> subvector('[1,2,3,4,5]'::vector, 1, 3) LIMIT 5;
Re-rank by the full vectors for better recall

SELECT * FROM (
    SELECT * FROM items ORDER BY subvector(embedding, 1, 3)::vector(3) <=> subvector('[1,2,3,4,5]'::vector, 1, 3) LIMIT 20
) ORDER BY embedding <=> '[1,2,3,4,5]' LIMIT 5;
Performance
Tuning
Use a tool like PgTune to set initial values for Postgres server parameters. For instance, shared_buffers should typically be 25% of the server’s memory. You can find the config file with:

SHOW config_file;
And check individual settings with:

SHOW shared_buffers;
Be sure to restart Postgres for changes to take effect.

Loading
Use COPY for bulk loading data (example).

COPY items (embedding) FROM STDIN WITH (FORMAT BINARY);
Add any indexes after loading the initial data for best performance.

Indexing
See index build time for HNSW and IVFFlat.

In production environments, create indexes concurrently to avoid blocking writes.

CREATE INDEX CONCURRENTLY ...
Querying
Use EXPLAIN ANALYZE to debug performance.

EXPLAIN ANALYZE SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
Exact Search
To speed up queries without an index, increase max_parallel_workers_per_gather.

SET max_parallel_workers_per_gather = 4;
If vectors are normalized to length 1 (like OpenAI embeddings), use inner product for best performance.

SELECT * FROM items ORDER BY embedding <#> '[3,1,2]' LIMIT 5;
Approximate Search
To speed up queries with an IVFFlat index, increase the number of inverted lists (at the expense of recall).

CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 1000);
Vacuuming
Vacuuming can take a while for HNSW indexes. Speed it up by reindexing first.

REINDEX INDEX CONCURRENTLY index_name;
VACUUM table_name;
Monitoring
Monitor performance with pg_stat_statements (be sure to add it to shared_preload_libraries).

CREATE EXTENSION pg_stat_statements;
Get the most time-consuming queries with:

SELECT query, calls, ROUND((total_plan_time + total_exec_time) / calls) AS avg_time_ms,
    ROUND((total_plan_time + total_exec_time) / 60000) AS total_time_min
    FROM pg_stat_statements ORDER BY total_plan_time + total_exec_time DESC LIMIT 20;
Note: Replace total_plan_time + total_exec_time with total_time for Postgres < 13

Monitor recall by comparing results from approximate search with exact search.

BEGIN;
SET LOCAL enable_indexscan = off; -- use exact search
SELECT ...
COMMIT;
Scaling
Scale pgvector the same way you scale Postgres.

Scale vertically by increasing memory, CPU, and storage on a single instance. Use existing tools to tune parameters and monitor performance.

Scale horizontally with replicas, or use Citus or another approach for sharding (example).

Languages
Use pgvector from any language with a Postgres client. You can even generate and store vectors in one language and query them in another.

Language	Libraries / Examples
C	pgvector-c
C++	pgvector-cpp
C#, F#, Visual Basic	pgvector-dotnet
Crystal	pgvector-crystal
D	pgvector-d
Dart	pgvector-dart
Elixir	pgvector-elixir
Erlang	pgvector-erlang
Fortran	pgvector-fortran
Gleam	pgvector-gleam
Go	pgvector-go
Haskell	pgvector-haskell
Java, Kotlin, Groovy, Scala	pgvector-java
JavaScript, TypeScript	pgvector-node
Julia	pgvector-julia
Lisp	pgvector-lisp
Lua	pgvector-lua
Nim	pgvector-nim
OCaml	pgvector-ocaml
Perl	pgvector-perl
PHP	pgvector-php
Python	pgvector-python
R	pgvector-r
Raku	pgvector-raku
Ruby	pgvector-ruby, Neighbor
Rust	pgvector-rust
Swift	pgvector-swift
Zig	pgvector-zig
Frequently Asked Questions
How many vectors can be stored in a single table?
A non-partitioned table has a limit of 32 TB by default in Postgres. A partitioned table can have thousands of partitions of that size.

Is replication supported?
Yes, pgvector uses the write-ahead log (WAL), which allows for replication and point-in-time recovery.

What if I want to index vectors with more than 2,000 dimensions?
You can use half-precision indexing to index up to 4,000 dimensions or binary quantization to index up to 64,000 dimensions. Another option is dimensionality reduction.

Can I store vectors with different dimensions in the same column?
You can use vector as the type (instead of vector(3)).

CREATE TABLE embeddings (model_id bigint, item_id bigint, embedding vector, PRIMARY KEY (model_id, item_id));
However, you can only create indexes on rows with the same number of dimensions (using expression and partial indexing):

CREATE INDEX ON embeddings USING hnsw ((embedding::vector(3)) vector_l2_ops) WHERE (model_id = 123);
and query with:

SELECT * FROM embeddings WHERE model_id = 123 ORDER BY embedding::vector(3) <-> '[3,1,2]' LIMIT 5;
Can I store vectors with more precision?
You can use the double precision[] or numeric[] type to store vectors with more precision.

CREATE TABLE items (id bigserial PRIMARY KEY, embedding double precision[]);

-- use {} instead of [] for Postgres arrays
INSERT INTO items (embedding) VALUES ('{1,2,3}'), ('{4,5,6}');
Optionally, add a check constraint to ensure data can be converted to the vector type and has the expected dimensions.

ALTER TABLE items ADD CHECK (vector_dims(embedding::vector) = 3);
Use expression indexing to index (at a lower precision):

CREATE INDEX ON items USING hnsw ((embedding::vector(3)) vector_l2_ops);
and query with:

SELECT * FROM items ORDER BY embedding::vector(3) <-> '[3,1,2]' LIMIT 5;
Do indexes need to fit into memory?
No, but like other index types, you’ll likely see better performance if they do. You can get the size of an index with:

SELECT pg_size_pretty(pg_relation_size('index_name'));
Troubleshooting
Why isn’t a query using an index?
The query needs to have an ORDER BY and LIMIT, and the ORDER BY must be the result of a distance operator (not an expression) in ascending order.

-- index
ORDER BY embedding <=> '[3,1,2]' LIMIT 5;

-- no index
ORDER BY 1 - (embedding <=> '[3,1,2]') DESC LIMIT 5;
You can encourage the planner to use an index for a query with:

BEGIN;
SET LOCAL enable_seqscan = off;
SELECT ...
COMMIT;
Also, if the table is small, a table scan may be faster.

Why isn’t a query using a parallel table scan?
The planner doesn’t consider out-of-line storage in cost estimates, which can make a serial scan look cheaper. You can reduce the cost of a parallel scan for a query with:

BEGIN;
SET LOCAL min_parallel_table_scan_size = 1;
SET LOCAL parallel_setup_cost = 1;
SELECT ...
COMMIT;
or choose to store vectors inline:

ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN;
Why are there less results for a query after adding an HNSW index?
Results are limited by the size of the dynamic candidate list (hnsw.ef_search), which is 40 by default. There may be even less results due to dead tuples or filtering conditions in the query. Enabling iterative index scans can help address this.

Also, note that NULL vectors are not indexed (as well as zero vectors for cosine distance).

Why are there less results for a query after adding an IVFFlat index?
The index was likely created with too little data for the number of lists. Drop the index until the table has more data.

DROP INDEX index_name;
Results can also be limited by the number of probes (ivfflat.probes). Enabling iterative index scans can address this.

Also, note that NULL vectors are not indexed (as well as zero vectors for cosine distance).

Reference
Vector
Halfvec
Bit
Sparsevec
Vector Type
Each vector takes 4 * dimensions + 8 bytes of storage. Each element is a single-precision floating-point number (like the real type in Postgres), and all elements must be finite (no NaN, Infinity or -Infinity). Vectors can have up to 16,000 dimensions.

Vector Operators
Operator	Description	Added
+	element-wise addition	
-	element-wise subtraction	
*	element-wise multiplication	0.5.0
||	concatenate	0.7.0
<->	Euclidean distance	
<#>	negative inner product	
<=>	cosine distance	
<+>	taxicab distance	0.7.0
Vector Functions
Function	Description	Added
binary_quantize(vector) → bit	binary quantize	0.7.0
cosine_distance(vector, vector) → double precision	cosine distance	
inner_product(vector, vector) → double precision	inner product	
l1_distance(vector, vector) → double precision	taxicab distance	0.5.0
l2_distance(vector, vector) → double precision	Euclidean distance	
l2_normalize(vector) → vector	Normalize with Euclidean norm	0.7.0
subvector(vector, integer, integer) → vector	subvector	0.7.0
vector_dims(vector) → integer	number of dimensions	
vector_norm(vector) → double precision	Euclidean norm	
Vector Aggregate Functions
Function	Description	Added
avg(vector) → vector	average	
sum(vector) → vector	sum	0.5.0
Halfvec Type
Each half vector takes 2 * dimensions + 8 bytes of storage. Each element is a half-precision floating-point number, and all elements must be finite (no NaN, Infinity or -Infinity). Half vectors can have up to 16,000 dimensions.

Halfvec Operators
Operator	Description	Added
+	element-wise addition	0.7.0
-	element-wise subtraction	0.7.0
*	element-wise multiplication	0.7.0
||	concatenate	0.7.0
<->	Euclidean distance	0.7.0
<#>	negative inner product	0.7.0
<=>	cosine distance	0.7.0
<+>	taxicab distance	0.7.0
Halfvec Functions
Function	Description	Added
binary_quantize(halfvec) → bit	binary quantize	0.7.0
cosine_distance(halfvec, halfvec) → double precision	cosine distance	0.7.0
inner_product(halfvec, halfvec) → double precision	inner product	0.7.0
l1_distance(halfvec, halfvec) → double precision	taxicab distance	0.7.0
l2_distance(halfvec, halfvec) → double precision	Euclidean distance	0.7.0
l2_norm(halfvec) → double precision	Euclidean norm	0.7.0
l2_normalize(halfvec) → halfvec	Normalize with Euclidean norm	0.7.0
subvector(halfvec, integer, integer) → halfvec	subvector	0.7.0
vector_dims(halfvec) → integer	number of dimensions	0.7.0
Halfvec Aggregate Functions
Function	Description	Added
avg(halfvec) → halfvec	average	0.7.0
sum(halfvec) → halfvec	sum	0.7.0
Bit Type
Each bit vector takes dimensions / 8 + 8 bytes of storage. See the Postgres docs for more info.

Bit Operators
Operator	Description	Added
<~>	Hamming distance	0.7.0
<%>	Jaccard distance	0.7.0
Bit Functions
Function	Description	Added
hamming_distance(bit, bit) → double precision	Hamming distance	0.7.0
jaccard_distance(bit, bit) → double precision	Jaccard distance	0.7.0
Sparsevec Type
Each sparse vector takes 8 * non-zero elements + 16 bytes of storage. Each element is a single-precision floating-point number, and all elements must be finite (no NaN, Infinity or -Infinity). Sparse vectors can have up to 16,000 non-zero elements.

Sparsevec Operators
Operator	Description	Added
<->	Euclidean distance	0.7.0
<#>	negative inner product	0.7.0
<=>	cosine distance	0.7.0
<+>	taxicab distance	0.7.0
Sparsevec Functions
Function	Description	Added
cosine_distance(sparsevec, sparsevec) → double precision	cosine distance	0.7.0
inner_product(sparsevec, sparsevec) → double precision	inner product	0.7.0
l1_distance(sparsevec, sparsevec) → double precision	taxicab distance	0.7.0
l2_distance(sparsevec, sparsevec) → double precision	Euclidean distance	0.7.0
l2_norm(sparsevec) → double precision	Euclidean norm	0.7.0
l2_normalize(sparsevec) → sparsevec	Normalize with Euclidean norm	0.7.0
Installation Notes - Linux and Mac
Postgres Location
If your machine has multiple Postgres installations, specify the path to pg_config with:

export PG_CONFIG=/Library/PostgreSQL/17/bin/pg_config
Then re-run the installation instructions (run make clean before make if needed). If sudo is needed for make install, use:

sudo --preserve-env=PG_CONFIG make install
A few common paths on Mac are:

EDB installer - /Library/PostgreSQL/17/bin/pg_config
Homebrew (arm64) - /opt/homebrew/opt/postgresql@17/bin/pg_config
Homebrew (x86-64) - /usr/local/opt/postgresql@17/bin/pg_config
Note: Replace 17 with your Postgres server version

Missing Header
If compilation fails with fatal error: postgres.h: No such file or directory, make sure Postgres development files are installed on the server.

For Ubuntu and Debian, use:

sudo apt install postgresql-server-dev-17
Note: Replace 17 with your Postgres server version

Missing SDK
If compilation fails and the output includes warning: no such sysroot directory on Mac, reinstall Xcode Command Line Tools.

Portability
By default, pgvector compiles with -march=native on some platforms for best performance. However, this can lead to Illegal instruction errors if trying to run the compiled extension on a different machine.

To compile for portability, use:

make OPTFLAGS=""
Installation Notes - Windows
Missing Header
If compilation fails with Cannot open include file: 'postgres.h': No such file or directory, make sure PGROOT is correct.

Permissions
If installation fails with Access is denied, re-run the installation instructions as an administrator.

Additional Installation Methods
Docker
Get the Docker image with:

docker pull pgvector/pgvector:pg17
This adds pgvector to the Postgres image (replace 17 with your Postgres server version, and run it the same way).

You can also build the image manually:

git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
docker build --pull --build-arg PG_MAJOR=17 -t myuser/pgvector .
Homebrew
With Homebrew Postgres, you can use:

brew install pgvector
Note: This only adds it to the postgresql@17 and postgresql@14 formulas

PGXN
Install from the PostgreSQL Extension Network with:

pgxn install vector
APT
Debian and Ubuntu packages are available from the PostgreSQL APT Repository. Follow the setup instructions and run:

sudo apt install postgresql-17-pgvector
Note: Replace 17 with your Postgres server version

Yum
RPM packages are available from the PostgreSQL Yum Repository. Follow the setup instructions for your distribution and run:

sudo yum install pgvector_17
# or
sudo dnf install pgvector_17
Note: Replace 17 with your Postgres server version

pkg
Install the FreeBSD package with:

pkg install postgresql16-pgvector
or the port with:

cd /usr/ports/databases/pgvector
make install
conda-forge
With Conda Postgres, install from conda-forge with:

conda install -c conda-forge pgvector
This method is community-maintained by @mmcauliffe

Postgres.app
Download the latest release with Postgres 15+.

Hosted Postgres
pgvector is available on these providers.

Upgrading
Install the latest version (use the same method as the original installation). Then in each database you want to upgrade, run:

ALTER EXTENSION vector UPDATE;
You can check the version in the current database with:

SELECT extversion FROM pg_extension WHERE extname = 'vector';
Upgrade Notes
0.6.0
Postgres 12
If upgrading with Postgres 12, remove this line from sql/vector--0.5.1--0.6.0.sql:

ALTER TYPE vector SET (STORAGE = external);
Then run make install and ALTER EXTENSION vector UPDATE;.

Docker
The Docker image is now published in the pgvector org, and there are tags for each supported version of Postgres (rather than a latest tag).

docker pull pgvector/pgvector:pg16
# or
docker pull pgvector/pgvector:0.6.0-pg16
Also, if you’ve increased maintenance_work_mem, make sure --shm-size is at least that size to avoid an error with parallel HNSW index builds.

docker run --shm-size=1g ...
Thanks
Thanks to:

PASE: PostgreSQL Ultra-High-Dimensional Approximate Nearest Neighbor Search Extension
Faiss: A Library for Efficient Similarity Search and Clustering of Dense Vectors
Using the Triangle Inequality to Accelerate k-means
k-means++: The Advantage of Careful Seeding
Concept Decompositions for Large Sparse Text Data using Clustering
Efficient and Robust Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graphs
History
View the changelog

Contributing
Everyone is encouraged to help improve this project. Here are a few ways you can help:

Report bugs
Fix bugs and submit pull requests
Write, clarify, or fix documentation
Suggest or add new features
To get started with development:

git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
To run all tests:

make installcheck        # regression tests
make prove_installcheck  # TAP tests
To run single tests:

make installcheck REGRESS=functions                            # regression test
make prove_installcheck PROVE_TESTS=test/t/001_ivfflat_wal.pl  # TAP test
To enable assertions:

make clean && PG_CFLAGS="-DUSE_ASSERT_CHECKING" make && make install
To enable benchmarking:

make clean && PG_CFLAGS="-DIVFFLAT_BENCH" make && make install
To show memory usage:

make clean && PG_CFLAGS="-DHNSW_MEMORY -DIVFFLAT_MEMORY" make && make install
To get k-means metrics:

make clean && PG_CFLAGS="-DIVFFLAT_KMEANS_DEBUG" make && make install
Resources for contributors

Extension Building Infrastructure
Index Access Method Interface Definition
Generic WAL Records

RAG with Permissions
Fine-grain access control with Retrieval Augmented Generation.
Since pgvector is built on top of Postgres, you can implement fine-grain access control on your vector database using Row Level Security (RLS). This means you can restrict which documents are returned during a vector similarity search to users that have access to them. Supabase also supports Foreign Data Wrappers (FDW) which means you can use an external database or data source to determine these permissions if your user data doesn't exist in Supabase.

Use this guide to learn how to restrict access to documents when performing retrieval augmented generation (RAG).

Example#
In a typical RAG setup, your documents are chunked into small subsections and similarity is performed over those sections:

-- Track documents/pages/files/etc
create table documents (
  id bigint primary key generated always as identity,
  name text not null,
  owner_id uuid not null references auth.users (id) default auth.uid(),
  created_at timestamp with time zone not null default now()
);

-- Store the content and embedding vector for each section in the document
-- with a reference to original document (one-to-many)
create table document_sections (
  id bigint primary key generated always as identity,
  document_id bigint not null references documents (id),
  content text not null,
  embedding vector (384)
);

Notice how we record the owner_id on each document. Let's create an RLS policy that restricts access to document_sections based on whether or not they own the linked document:

-- enable row level security
alter table document_sections enable row level security;

-- setup RLS for select operations
create policy "Users can query their own document sections"
on document_sections for select to authenticated using (
  document_id in (
    select id
    from documents
    where (owner_id = (select auth.uid()))
  )
);

In this example, the current user is determined using the built-in auth.uid() function when the query is executed through your project's auto-generated REST API. If you are connecting to your Supabase database through a direct Postgres connection, see Direct Postgres Connection below for directions on how to achieve the same access control.

Now every select query executed on document_sections will implicitly filter the returned sections based on whether or not the current user has access to them.

For example, executing:

select * from document_sections;

as an authenticated user will only return rows that they are the owner of (as determined by the linked document). More importantly, semantic search over these sections (or any additional filtering for that matter) will continue to respect these RLS policies:

-- Perform inner product similarity based on a match_threshold
select *
from document_sections
where document_sections.embedding <#> embedding < -match_threshold
order by document_sections.embedding <#> embedding;

The above example only configures select access to users. If you wanted, you could create more RLS policies for inserts, updates, and deletes in order to apply the same permission logic for those other operations. See Row Level Security for a more in-depth guide on RLS policies.

Alternative scenarios#
Every app has its own unique requirements and may differ from the above example. Here are some alternative scenarios we often see and how they are implemented in Supabase.

Documents owned by multiple people#
Instead of a one-to-many relationship between users and documents, you may require a many-to-many relationship so that multiple people can access the same document. Let's reimplement this using a join table:

create table document_owners (
  id bigint primary key generated always as identity,
  owner_id uuid not null references auth.users (id) default auth.uid(),
  document_id bigint not null references documents (id)
);

Then your RLS policy would change to:

create policy "Users can query their own document sections"
on document_sections for select to authenticated using (
  document_id in (
    select document_id
    from document_owners
    where (owner_id = (select auth.uid()))
  )
);

Instead of directly querying the documents table, we query the join table.

User and document data live outside of Supabase#
You may have an existing system that stores users, documents, and their permissions in a separate database. Let's explore the scenario where this data exists in another Postgres database. We'll use a foreign data wrapper (FDW) to connect to the external DB from within your Supabase DB:

RLS is latency-sensitive, so extra caution should be taken before implementing this method. Use the query plan analyzer to measure execution times for your queries to ensure they are within expected ranges. For enterprise applications, contact enterprise@supabase.io.

For data sources other than Postgres, see Foreign Data Wrappers for a list of external sources supported today. If your data lives in a source not provided in the list, please contact support and we'll be happy to discuss your use case.

Let's assume your external DB contains a users and documents table like this:

create table public.users (
  id bigint primary key generated always as identity,
  email text not null,
  created_at timestamp with time zone not null default now()
);

create table public.documents (
  id bigint primary key generated always as identity,
  name text not null,
  owner_id bigint not null references public.users (id),
  created_at timestamp with time zone not null default now()
);

In your Supabase DB, let's create foreign tables that link to the above tables:

create schema external;
create extension postgres_fdw with schema extensions;

-- Setup the foreign server
create server foreign_server
  foreign data wrapper postgres_fdw
  options (host '<db-host>', port '<db-port>', dbname '<db-name>');

-- Map local 'authenticated' role to external 'postgres' user
create user mapping for authenticated
  server foreign_server
  options (user 'postgres', password '<user-password>');

-- Import foreign 'users' and 'documents' tables into 'external' schema
import foreign schema public limit to (users, documents)
  from server foreign_server into external;

This example maps the authenticated role in Supabase to the postgres user in the external DB. In production, it's best to create a custom user on the external DB that has the minimum permissions necessary to access the information you need.

On the Supabase DB, we use the built-in authenticated role which is automatically used when end users make authenticated requests over your auto-generated REST API. If you plan to connect to your Supabase DB over a direct Postgres connection instead of the REST API, you can change this to any user you like. See Direct Postgres Connection for more info.

We'll store document_sections and their embeddings in Supabase so that we can perform similarity search over them via pgvector.

create table document_sections (
  id bigint primary key generated always as identity,
  document_id bigint not null,
  content text not null,
  embedding vector (384)
);

We maintain a reference to the foreign document via document_id, but without a foreign key reference since foreign keys can only be added to local tables. Be sure to use the same ID data type that you use on your external documents table.

Since we're managing users and authentication outside of Supabase, we have two options:

Make a direct Postgres connection to the Supabase DB and set the current user every request
Issue a custom JWT from your system and use it to authenticate with the REST API
Direct Postgres connection#
You can directly connect to your Supabase Postgres DB using the connection info on your project's database settings page. To use RLS with this method, we use a custom session variable that contains the current user's ID:

-- enable row level security
alter table document_sections enable row level security;

-- setup RLS for select operations
create policy "Users can query their own document sections"
on document_sections for select to authenticated using (
  document_id in (
    select id
    from external.documents
    where owner_id = current_setting('app.current_user_id')::bigint
  )
);

The session variable is accessed through the current_setting() function. We name the variable app.current_user_id here, but you can modify this to any name you like. We also cast it to a bigint since that was the data type of the user.id column. Change this to whatever data type you use for your ID.

Now for every request, we set the user's ID at the beginning of the session:

set app.current_user_id = '<current-user-id>';

Then all subsequent queries will inherit the permission of that user:

-- Only document sections owned by the user are returned
select *
from document_sections
where document_sections.embedding <#> embedding < -match_threshold
order by document_sections.embedding <#> embedding;

You might be tempted to discard RLS completely and simply filter by user within the where clause. Though this will work, we recommend RLS as a general best practice since RLS is always applied even as new queries and application logic is introduced in the future.

Custom JWT with REST API#
If you would like to use the auto-generated REST API to query your Supabase database using JWTs from an external auth provider, you can get your auth provider to issue a custom JWT for Supabase.

See the Clerk Supabase docs for an example of how this can be done. Modify the instructions to work with your own auth provider as needed.

Now we can simply use the same RLS policy from our first example:

-- enable row level security
alter table document_sections enable row level security;

-- setup RLS for select operations
create policy "Users can query their own document sections"
on document_sections for select to authenticated using (
  document_id in (
    select id
    from documents
    where (owner_id = (select auth.uid()))
  )
);

Under the hood, auth.uid() references current_setting('request.jwt.claim.sub') which corresponds to the JWT's sub (subject) claim. This setting is automatically set at the beginning of each request to the REST API.

All subsequent queries will inherit the permission of that user:

-- Only document sections owned by the user are returned
select *
from document_sections
where document_sections.embedding <#> embedding < -match_threshold
order by document_sections.embedding <#> embedding;

Other scenarios#
There are endless approaches to this problem based on the complexities of each system. Luckily Postgres comes with all the primitives needed to provide access control in the way that works best for your project.

If the examples above didn't fit your use case or you need to adjust them slightly to better fit your existing system, feel free to reach out to support and we'll be happy to assist you.

Semantic search
Learn how to search by meaning rather than exact keywords.
Semantic search interprets the meaning behind user queries rather than exact keywords. It uses machine learning to capture the intent and context behind the query, handling language nuances like synonyms, phrasing variations, and word relationships.

When to use semantic search#
Semantic search is useful in applications where the depth of understanding and context is important for delivering relevant results. A good example is in customer support or knowledge base search engines. Users often phrase their problems or questions in various ways, and a traditional keyword-based search might not always retrieve the most helpful documents. With semantic search, the system can understand the meaning behind the queries and match them with relevant solutions or articles, even if the exact wording differs.

For instance, a user searching for "increase text size on display" might miss articles titled "How to adjust font size in settings" in a keyword-based search system. However, a semantic search engine would understand the intent behind the query and correctly match it to relevant articles, regardless of the specific terminology used.

It's also possible to combine semantic search with keyword search to get the best of both worlds. See Hybrid search for more details.

How semantic search works#
Semantic search uses an intermediate representation called an “embedding vector” to link database records with search queries. A vector, in the context of semantic search, is a list of numerical values. They represent various features of the text and allow for the semantic comparison between different pieces of text.

The best way to think of embeddings is by plotting them on a graph, where each embedding is a single point whose coordinates are the numerical values within its vector. Importantly, embeddings are plotted such that similar concepts are positioned close together while dissimilar concepts are far apart. For more details, see What are embeddings?

Embeddings are generated using a language model, and embeddings are compared to each other using a similarity metric. The language model is trained to understand the semantics of language, including syntax, context, and the relationships between words. It generates embeddings for both the content in the database and the search queries. Then the similarity metric, often a function like cosine similarity or dot product, is used to compare the query embeddings with the document embeddings (in other words, to measure how close they are to each other on the graph). The documents with embeddings most similar to the query's are deemed the most relevant and are returned as search results.

Embedding models#
There are many embedding models available today. Supabase Edge Functions has built in support for the gte-small model. Others can be accessed through third-party APIs like OpenAI, where you send your text in the request and receive an embedding vector in the response. Others can run locally on your own compute, such as through Transformers.js for JavaScript implementations. For more information on local implementation, see Generate embeddings.

It's crucial to remember that when using embedding models with semantic search, you must use the same model for all embedding comparisons. Comparing embeddings created by different models will yield meaningless results.

Semantic search in Postgres#
To implement semantic search in Postgres we use pgvector - an extension that allows for efficient storage and retrieval of high-dimensional vectors. These vectors are numerical representations of text (or other types of data) generated by embedding models.

Enable the pgvector extension by running:

create extension vector
with
  schema extensions;

Create a table to store the embeddings:

create table documents (
  id bigint primary key generated always as identity,
  content text,
  embedding vector(512)
);

Or if you have an existing table, you can add a vector column like so:

alter table documents
add column embedding vector(512);

In this example, we create a column named embedding which uses the newly enabled vector data type. The size of the vector (as indicated in parentheses) represents the number of dimensions in the embedding. Here we use 512, but adjust this to match the number of dimensions produced by your embedding model.

For more details on vector columns, including how to generate embeddings and store them, see Vector columns.

Similarity metric#
pgvector support 3 operators for computing distance between embeddings:

Operator	Description
<->	Euclidean distance
<#>	negative inner product
<=>	cosine distance
These operators are used directly in your SQL query to retrieve records that are most similar to the user's search query. Choosing the right operator depends on your needs. Inner product (also known as dot product) tends to be the fastest if your vectors are normalized.

The easiest way to perform semantic search in Postgres is by creating a function:

-- Match documents using cosine distance (<=>)
create or replace function match_documents (
  query_embedding vector(512),
  match_threshold float,
  match_count int
)
returns setof documents
language sql
as $$
  select *
  from documents
  where documents.embedding <=> query_embedding < 1 - match_threshold
  order by documents.embedding <=> query_embedding asc
  limit least(match_count, 200);
$$;

Here we create a function match_documents that accepts three parameters:

query_embedding: a one-time embedding generated for the user's search query. Here we set the size to 512, but adjust this to match the number of dimensions produced by your embedding model.
match_threshold: the minimum similarity between embeddings. This is a value between 1 and -1, where 1 is most similar and -1 is most dissimilar.
match_count: the maximum number of results to return. Note the query may return less than this number if match_threshold resulted in a small shortlist. Limited to 200 records to avoid unintentionally overloading your database.
In this example, we return a setof documents and refer to documents throughout the query. Adjust this to use the relevant tables in your application.

You'll notice we are using the cosine distance (<=>) operator in our query. Cosine distance is a safe default when you don't know whether or not your embeddings are normalized. If you know for a fact that they are normalized (for example, your embedding is returned from OpenAI), you can use negative inner product (<#>) for better performance:

-- Match documents using negative inner product (<#>)
create or replace function match_documents (
  query_embedding vector(512),
  match_threshold float,
  match_count int
)
returns setof documents
language sql
as $$
  select *
  from documents
  where documents.embedding <#> query_embedding < -match_threshold
  order by documents.embedding <#> query_embedding asc
  limit least(match_count, 200);
$$;

Note that since <#> is negative, we negate match_threshold accordingly in the where clause. For more information on the different operators, see the pgvector docs.

Calling from your application#
Finally you can execute this function from your application. If you are using a Supabase client library such as supabase-js, you can invoke it using the rpc() method:

const { data: documents } = await supabase.rpc('match_documents', {
  query_embedding: embedding, // pass the query embedding
  match_threshold: 0.78, // choose an appropriate threshold for your data
  match_count: 10, // choose the number of matches
})

You can also call this method directly from SQL:

select *
from match_documents(
  '[...]'::vector(512), -- pass the query embedding
  0.78, -- chose an appropriate threshold for your data
  10 -- choose the number of matches
);

In this scenario, you'll likely use a Postgres client library to establish a direct connection from your application to the database. It's best practice to parameterize your arguments before executing the query.

Next steps#
As your database scales, you will need an index on your vector columns to maintain fast query speeds. See Vector indexes for an in-depth guide on the different types of indexes and how they work.

Keyword search
Learn how to search by words or phrases.
Keyword search involves locating documents or records that contain specific words or phrases, primarily based on the exact match between the search terms and the text within the data. It differs from semantic search, which interprets the meaning behind the query to provide results that are contextually related, even if the exact words aren't present in the text. Semantic search considers synonyms, intent, and natural language nuances to provide a more nuanced approach to information retrieval.

In Postgres, keyword search is implemented using full-text search. It supports indexing and text analysis for data retrieval, focusing on records that match the search criteria. Postgres' full-text search extends beyond simple keyword matching to address linguistic nuances, making it effective for applications that require precise text queries.

Why would I want to use keyword search?#
Keyword search is particularly useful in scenarios where precision and specificity matter. It's more effective than semantic search when users are looking for information using exact terminology or specific identifiers. It ensures that results directly contain those terms, reducing the chance of retrieving irrelevant information that might be semantically related but not what the user seeks.

For example in technical or academic research databases, researchers often search for specific studies, compounds, or concepts identified by certain terms or codes. Searching for a specific chemical compound using its exact molecular formula or a unique identifier will yield more focused and relevant results compared to a semantic search, which could return a wide range of documents discussing the compound in different contexts. Keyword search ensures documents that explicitly mention the exact term are found, allowing users to access the precise data they need efficiently.

It's also possible to combine keyword search with semantic search to get the best of both worlds. See Hybrid search for more details.

Using full-text search#
For an in-depth guide to Postgres' full-text search, including how to store, index, and query records, see Full text search.

Hybrid search
Combine keyword search with semantic search.
Hybrid search combines full text search (searching by keyword) with semantic search (searching by meaning) to identify results that are both directly and contextually relevant to the user's query.

Why would I want to use hybrid search?#
Sometimes a single search method doesn't quite capture what a user is really looking for. For example, if a user searches for "Italian recipes with tomato sauce" on a cooking app, a keyword search would pull up recipes that specifically mention "Italian," "recipes," and "tomato sauce" in the text. However, it might miss out on dishes that are quintessentially Italian and use tomato sauce but don't explicitly label themselves with these words, or use variations like "pasta sauce" or "marinara." On the other hand, a semantic search might understand the culinary context and find recipes that match the intent, such as a traditional "Spaghetti Marinara," even if they don't match the exact keyword phrase. However, it could also suggest recipes that are contextually related but not what the user is looking for, like a "Mexican salsa" recipe, because it understands the context to be broadly about tomato-based sauces.

Hybrid search combines the strengths of both these methods. It would ensure that recipes explicitly mentioning the keywords are prioritized, thus capturing direct hits that satisfy the keyword criteria. At the same time, it would include recipes identified through semantic understanding as being related in meaning or context, like different Italian dishes that traditionally use tomato sauce but might not have been tagged explicitly with the user's search terms. It identifies results that are both directly and contextually relevant to the user's query while ideally minimizing misses and irrelevant suggestions.

When would I want to use hybrid search?#
The decision to use hybrid search depends on what your users are looking for in your app. For a code repository where developers need to find exact lines of code or error messages, keyword search is likely ideal because it matches specific terms. In a mental health forum where users search for advice or experiences related to their feelings, semantic search may be better because it finds results based on the meaning of a query, not just specific words. For a shopping app where customers might search for specific product names yet also be open to related suggestions, hybrid search combines the best of both worlds - finding exact matches while also uncovering similar products based on the shopping context.

How to combine search methods#
Hybrid search merges keyword search and semantic search, but how does this process work?

First, each search method is executed separately. Keyword search, which involves searching by specific words or phrases present in the content, will yield its own set of results. Similarly, semantic search, which involves understanding the context or meaning behind the search query rather than the specific words used, will generate its own unique results.

Now with these separate result lists available, the next step is to combine them into a single, unified list. This is achieved through a process known as “fusion”. Fusion takes the results from both search methods and merges them together based on a certain ranking or scoring system. This system may prioritize certain results based on factors like their relevance to the search query, their ranking in the individual lists, or other criteria. The result is a final list that integrates the strengths of both keyword and semantic search methods.

Reciprocal Ranked Fusion (RRF)#
One of the most common fusion methods is Reciprocal Ranked Fusion (RRF). The key idea behind RRF is to give more weight to the top-ranked items in each individual result list when building the final combined list.

In RRF, we iterate over each record and assign a score (noting that each record could exist in one or both lists). The score is calculated as 1 divided by that record's rank in each list, summed together between both lists. For example, if a record with an ID of 123 was ranked third in the keyword search and ninth in semantic search, it would receive a score of 
1
3
+
1
9
=
0.444
3
1
​
 + 
9
1
​
 =0.444. If the record was found in only one list and not the other, it would receive a score of 0 for the other list. The records are then sorted by this score to create the final list. The items with the highest scores are ranked first, and lowest scores ranked last.

This method ensures that items that are ranked high in multiple lists are given a high rank in the final list. It also ensures that items that are ranked high in only a few lists but low in others are not given a high rank in the final list. Placing the rank in the denominator when calculating score helps penalize the low ranking records.

Smoothing constant k#
To prevent extremely high scores for items that are ranked first (since we're dividing by the rank), a k constant is often added to the denominator to smooth the score:

1
k
+
r
a
n
k
k+rank
1
​
 

This constant can be any positive number, but is typically small. A constant of 1 would mean that a record ranked first would have a score of 
1
1
+
1
=
0.5
1+1
1
​
 =0.5 instead of 
1
1. This adjustment can help balance the influence of items that are ranked very high in individual lists when creating the final combined list.

Hybrid search in Postgres#
Let's implement hybrid search in Postgres using tsvector (keyword search) and pgvector (semantic search).

First we'll create a documents table to store the documents that we will search over. This is just an example - adjust this to match the structure of your application.

create table documents (
  id bigint primary key generated always as identity,
  content text,
  fts tsvector generated always as (to_tsvector('english', content)) stored,
  embedding vector(512)
);

The table contains 4 columns:

id is an auto-generated unique ID for the record. We'll use this later to match records when performing RRF.
content contains the actual text we will be searching over.
fts is an auto-generated tsvector column that is generated using the text in content. We will use this for full text search (search by keyword).
embedding is a vector column that stores the vector generated from our embedding model. We will use this for semantic search (search by meaning). We chose 512 dimensions for this example, but adjust this to match the size of the embedding vectors generated from your preferred model.
Next we'll create indexes on the fts and embedding columns so that their individual queries will remain fast at scale:

-- Create an index for the full-text search
create index on documents using gin(fts);

-- Create an index for the semantic vector search
create index on documents using hnsw (embedding vector_ip_ops);

For full text search we use a generalized inverted (GIN) index which is designed for handling composite values like those stored in a tsvector.

For semantic vector search we use an HNSW index, which is a high performing approximate nearest neighbor (ANN) search algorithm. Note that we are using the vector_ip_ops (inner product) operator with this index because we plan on using the inner product (<#>) operator later in our query. If you plan to use a different operator like cosine distance (<=>), be sure to update the index accordingly. For more information, see distance operators.

Finally we'll create our hybrid_search function:

create or replace function hybrid_search(
  query_text text,
  query_embedding vector(512),
  match_count int,
  full_text_weight float = 1,
  semantic_weight float = 1,
  rrf_k int = 50
)
returns setof documents
language sql
as $$
with full_text as (
  select
    id,
    -- Note: ts_rank_cd is not indexable but will only rank matches of the where clause
    -- which shouldn't be too big
    row_number() over(order by ts_rank_cd(fts, websearch_to_tsquery(query_text)) desc) as rank_ix
  from
    documents
  where
    fts @@ websearch_to_tsquery(query_text)
  order by rank_ix
  limit least(match_count, 30) * 2
),
semantic as (
  select
    id,
    row_number() over (order by embedding <#> query_embedding) as rank_ix
  from
    documents
  order by rank_ix
  limit least(match_count, 30) * 2
)
select
  documents.*
from
  full_text
  full outer join semantic
    on full_text.id = semantic.id
  join documents
    on coalesce(full_text.id, semantic.id) = documents.id
order by
  coalesce(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
  coalesce(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight
  desc
limit
  least(match_count, 30)
$$;

Let's break this down:

Parameters: The function accepts quite a few parameters, but the main (required) ones are query_text, query_embedding, and match_count.

query_text is the user's query text (more on this shortly)
query_embedding is the vector representation of the user's query produced by the embedding model. We chose 512 dimensions for this example, but adjust this to match the size of the embedding vectors generated from your preferred model. This must match the size of the embedding vector on the documents table (and use the same model).
match_count is the number of records returned in the limit clause.
The other parameters are optional, but give more control over the fusion process.

full_text_weight and semantic_weight decide how much weight each search method gets in the final score. These are both 1 by default which means they both equally contribute towards the final rank. A full_text_weight of 2 and semantic_weight of 1 would give full-text search twice as much weight as semantic search.
rrf_k is the k smoothing constant added to the reciprocal rank. The default is 50.
Return type: The function returns a set of records from our documents table.

CTE: We create two common table expressions (CTE), one for full-text search and one for semantic search. These perform each query individually prior to joining them.

RRF: The final query combines the results from the two CTEs using reciprocal rank fusion (RRF).

Running hybrid search#
To use this function in SQL, we can run:

select
  *
from
  hybrid_search(
    'Italian recipes with tomato sauce', -- user query
    '[...]'::vector(512), -- embedding generated from user query
    10
  );

In practice, you will likely be calling this from the Supabase client or through a custom backend layer. Here is a quick example of how you might call this from an Edge Function using JavaScript:

import { createClient } from 'jsr:@supabase/supabase-js@2'
import OpenAI from 'npm:openai'

const supabaseUrl = Deno.env.get('SUPABASE_URL')!
const supabaseServiceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
const openaiApiKey = Deno.env.get('OPENAI_API_KEY')!

Deno.serve(async (req) => {
  // Grab the user's query from the JSON payload
  const { query } = await req.json()

  // Instantiate OpenAI client
  const openai = new OpenAI({ apiKey: openaiApiKey })

  // Generate a one-time embedding for the user's query
  const embeddingResponse = await openai.embeddings.create({
    model: 'text-embedding-3-large',
    input: query,
    dimensions: 512,
  })

  const [{ embedding }] = embeddingResponse.data

  // Instantiate the Supabase client
  // (replace service role key with user's JWT if using Supabase auth and RLS)
  const supabase = createClient(supabaseUrl, supabaseServiceRoleKey)

  // Call hybrid_search Postgres function via RPC
  const { data: documents } = await supabase.rpc('hybrid_search', {
    query_text: query,
    query_embedding: embedding,
    match_count: 10,
  })

  return new Response(JSON.stringify(documents), {
    headers: { 'Content-Type': 'application/json' },
  })
})

This uses OpenAI's text-embedding-3-large model to generate embeddings (shortened to 512 dimensions for faster retrieval). Swap in your preferred embedding model (and dimension size) accordingly.

To test this, make a POST request to the function's endpoint while passing in a JSON payload containing the user's query. Here is an example POST request using cURL:

curl -i --location --request POST \
  'http://127.0.0.1:54321/functions/v1/hybrid-search' \
  --header 'Authorization: Bearer <anonymous key>' \
  --header 'Content-Type: application/json' \
  --data '{"query":"Italian recipes with tomato sauce"}'

For more information on how to create, test, and deploy edge functions, see Getting started.