-- Enable pgvector
create extension if not exists vector;

-- Products table
create table if not exists products (
    id            text primary key,
    title         text not null,
    description   text default '',
    product_type  text default '',
    tags          text[] default '{}',
    price         numeric default 0,
    currency      text default 'IDR',
    inventory_qty integer default 0,
    in_stock      boolean default false,
    image_url     text default '',
    url           text default '',
    embedding     vector(384),
    updated_at    timestamptz default now()
);

-- HNSW index for cosine similarity
create index if not exists products_embedding_idx
    on products using hnsw (embedding vector_cosine_ops);

-- Hybrid search RPC: cosine similarity + metadata filters
create or replace function match_products(
    query_embedding vector(384),
    match_count int default 6,
    min_price numeric default null,
    max_price numeric default null,
    category text default null,
    only_in_stock boolean default false
)
returns table (
    id text,
    title text,
    description text,
    product_type text,
    price numeric,
    currency text,
    in_stock boolean,
    image_url text,
    url text,
    similarity float
)
language sql stable
as $$
    select
        p.id, p.title, p.description, p.product_type,
        p.price, p.currency, p.in_stock, p.image_url, p.url,
        1 - (p.embedding <=> query_embedding) as similarity
    from products p
    where (min_price is null or p.price >= min_price)
      and (max_price is null or p.price <= max_price)
      and (category is null or p.product_type ilike '%' || category || '%')
      and (only_in_stock = false or p.in_stock = true)
    order by p.embedding <=> query_embedding
    limit match_count;
$$;
