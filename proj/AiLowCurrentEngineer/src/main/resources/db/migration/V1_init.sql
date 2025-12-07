create table projects (
                          id uuid primary key,
                          name varchar(200) not null,
                          src_key text,
                          created_at timestamptz not null default now(),
                          updated_at timestamptz not null default now()
);

create type job_status as enum ('PENDING','RUNNING','DONE','ERROR');

create table jobs (
                      id uuid primary key,
                      project_id uuid not null references projects(id) on delete cascade,
                      status job_status not null,
                      preferences_text text,
                      total_fixtures int,
                      target_lux double precision,
                      efficacy_lm_per_w double precision,
                      maintenance_factor double precision,
                      utilization_factor double precision,
                      export_formats text,          -- JSON-string (e.g. '["PDF","DXF"]')
                      parsed_json text,             -- JSON from /infer
                      lighting_json text,           -- JSON from /infer
                      exported_files_json text,     -- JSON from /infer
                      uploaded_uris_json text,      -- JSON from /infer
                      error_text text,
                      created_at timestamptz not null default now(),
                      updated_at timestamptz not null default now()
);

create index idx_jobs_project_id on jobs(project_id);
