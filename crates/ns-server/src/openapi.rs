//! Auto-generated OpenAPI 3.1 specification for the NextStat server.
//!
//! Served at `GET /v1/openapi.json`.

use serde_json::{Value, json};

pub fn openapi_spec() -> Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "NextStat Inference API",
            "description": "GPU-accelerated statistical inference server for HEP, pharma, and general-purpose fitting.",
            "version": ns_core::VERSION,
            "license": {
                "name": "BSL-1.1 / Commercial",
                "url": "https://nextstat.io/license"
            }
        },
        "servers": [
            { "url": "http://localhost:3742", "description": "Local dev server" }
        ],
        "security": [
            { "BearerAuth": [] }
        ],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "API key passed as Bearer token. Disabled when --api-keys is not set."
                }
            },
            "schemas": {
                "GpuSelector": {
                    "oneOf": [
                        { "type": "boolean" },
                        { "type": "string", "enum": ["cuda", "metal", "cpu", "auto"] }
                    ],
                    "default": true,
                    "description": "GPU device selector. true=auto, false/cpu=CPU only, or explicit device."
                },
                "FitRequest": {
                    "type": "object",
                    "required": ["workspace"],
                    "properties": {
                        "workspace": { "type": "object", "description": "pyhf or HS3 workspace JSON" },
                        "model_id": { "type": "string", "description": "Cached model ID (SHA-256)" },
                        "gpu": { "$ref": "#/components/schemas/GpuSelector" }
                    }
                },
                "FitResponse": {
                    "type": "object",
                    "properties": {
                        "parameter_names": { "type": "array", "items": { "type": "string" } },
                        "poi_index": { "type": "integer", "nullable": true },
                        "bestfit": { "type": "array", "items": { "type": "number" } },
                        "uncertainties": { "type": "array", "items": { "type": "number" } },
                        "nll": { "type": "number" },
                        "twice_nll": { "type": "number" },
                        "converged": { "type": "boolean" },
                        "n_iter": { "type": "integer" },
                        "n_fev": { "type": "integer" },
                        "n_gev": { "type": "integer" },
                        "covariance": { "type": "array", "items": { "type": "number" }, "nullable": true },
                        "device": { "type": "string" },
                        "wall_time_s": { "type": "number" }
                    }
                },
                "UnbinnedFitRequest": {
                    "type": "object",
                    "required": ["spec"],
                    "properties": {
                        "spec": { "type": "object", "description": "Unbinned spec JSON (nextstat_unbinned_spec_v0 schema)" },
                        "data_root": { "type": "string", "default": ".", "description": "Server-side dir for data files" }
                    }
                },
                "NlmeFitRequest": {
                    "type": "object",
                    "required": ["model_type", "times", "observations", "dose", "sigma"],
                    "properties": {
                        "model_type": { "type": "string", "enum": ["pk_1cpt", "nlme_1cpt"] },
                        "times": { "type": "array", "items": { "type": "number" } },
                        "observations": { "type": "array", "items": { "type": "number" } },
                        "dose": { "type": "number" },
                        "bioavailability": { "type": "number", "default": 1.0 },
                        "sigma": { "type": "number" },
                        "subject_idx": { "type": "array", "items": { "type": "integer" }, "description": "Required for nlme_1cpt" },
                        "n_subjects": { "type": "integer", "description": "Required for nlme_1cpt" },
                        "lloq": { "type": "number", "nullable": true },
                        "lloq_policy": { "type": "string", "enum": ["ignore", "replace_half", "censored"], "default": "censored" }
                    }
                },
                "BatchToysRequest": {
                    "type": "object",
                    "required": ["workspace"],
                    "properties": {
                        "workspace": { "type": "object" },
                        "params": { "type": "array", "items": { "type": "number" }, "nullable": true },
                        "n_toys": { "type": "integer", "default": 1000 },
                        "seed": { "type": "integer", "default": 42 },
                        "gpu": { "$ref": "#/components/schemas/GpuSelector" }
                    }
                },
                "JobSubmitRequest": {
                    "type": "object",
                    "required": ["task_type", "payload"],
                    "properties": {
                        "task_type": { "type": "string", "enum": ["batch_toys"] },
                        "payload": { "type": "object", "description": "Task-specific payload (same schema as sync endpoint)" }
                    }
                },
                "JobStatus": {
                    "type": "string",
                    "enum": ["pending", "running", "completed", "failed", "cancelled"]
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": { "type": "string" }
                    }
                }
            }
        },
        "paths": {
            "/v1/fit": {
                "post": {
                    "summary": "MLE fit (HistFactory binned model)",
                    "tags": ["Inference"],
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/FitRequest" } } } },
                    "responses": {
                        "200": { "description": "Fit result", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/FitResponse" } } } },
                        "400": { "description": "Bad request", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
                        "401": { "description": "Unauthorized" },
                        "429": { "description": "Rate limited" },
                        "500": { "description": "Internal error" }
                    }
                }
            },
            "/v1/ranking": {
                "post": {
                    "summary": "Systematic ranking (impact plot)",
                    "tags": ["Inference"],
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/FitRequest" } } } },
                    "responses": {
                        "200": { "description": "Ranked systematics" },
                        "400": { "description": "Bad request" }
                    }
                }
            },
            "/v1/batch/fit": {
                "post": {
                    "summary": "Batch MLE fit (multiple workspaces)",
                    "tags": ["Batch"],
                    "requestBody": { "required": true, "content": { "application/json": { "schema": {
                        "type": "object",
                        "required": ["workspaces"],
                        "properties": {
                            "workspaces": { "type": "array", "items": { "type": "object" }, "maxItems": 100 },
                            "gpu": { "$ref": "#/components/schemas/GpuSelector" }
                        }
                    } } } },
                    "responses": { "200": { "description": "Array of fit results" } }
                }
            },
            "/v1/batch/toys": {
                "post": {
                    "summary": "Batch toy fits (pseudo-experiments)",
                    "tags": ["Batch"],
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/BatchToysRequest" } } } },
                    "responses": { "200": { "description": "Toy fit results" } }
                }
            },
            "/v1/unbinned/fit": {
                "post": {
                    "summary": "Unbinned MLE fit (event-level likelihood)",
                    "tags": ["Pharma / Unbinned"],
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/UnbinnedFitRequest" } } } },
                    "responses": {
                        "200": { "description": "Fit result", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/FitResponse" } } } },
                        "400": { "description": "Bad request" }
                    }
                }
            },
            "/v1/nlme/fit": {
                "post": {
                    "summary": "NLME / PK population fit",
                    "tags": ["Pharma / NLME"],
                    "description": "Supports pk_1cpt (individual) and nlme_1cpt (population NLME with log-normal random effects).",
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/NlmeFitRequest" } } } },
                    "responses": { "200": { "description": "NLME fit result" }, "400": { "description": "Bad request" } }
                }
            },
            "/v1/jobs/submit": {
                "post": {
                    "summary": "Submit async job",
                    "tags": ["Jobs"],
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/JobSubmitRequest" } } } },
                    "responses": { "200": { "description": "Job ID + pending status" } }
                }
            },
            "/v1/jobs/{id}": {
                "get": {
                    "summary": "Poll job status",
                    "tags": ["Jobs"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": { "200": { "description": "Job status + result if completed" }, "404": { "description": "Job not found" } }
                },
                "delete": {
                    "summary": "Cancel a job",
                    "tags": ["Jobs"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": { "200": { "description": "Cancellation confirmed" } }
                }
            },
            "/v1/jobs": {
                "get": {
                    "summary": "List all jobs",
                    "tags": ["Jobs"],
                    "responses": { "200": { "description": "Array of job statuses" } }
                }
            },
            "/v1/tools/schema": {
                "get": {
                    "summary": "OpenAI function-calling tool schema",
                    "tags": ["Tools"],
                    "responses": { "200": { "description": "JSON schema for all supported tools" } }
                }
            },
            "/v1/tools/execute": {
                "post": {
                    "summary": "Execute a tool (OpenAI function-calling)",
                    "tags": ["Tools"],
                    "responses": { "200": { "description": "Tool execution result" } }
                }
            },
            "/v1/models": {
                "post": {
                    "summary": "Upload workspace to model cache",
                    "tags": ["Models"],
                    "responses": { "200": { "description": "Model ID + metadata" } }
                },
                "get": {
                    "summary": "List cached models",
                    "tags": ["Models"],
                    "responses": { "200": { "description": "Array of cached model IDs" } }
                }
            },
            "/v1/models/{id}": {
                "delete": {
                    "summary": "Delete cached model",
                    "tags": ["Models"],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": { "200": { "description": "Deletion confirmed" } }
                }
            },
            "/v1/health": {
                "get": {
                    "summary": "Server health check",
                    "tags": ["Admin"],
                    "security": [],
                    "responses": { "200": { "description": "Server status, version, GPU info, uptime" } }
                }
            },
            "/v1/openapi.json": {
                "get": {
                    "summary": "OpenAPI specification",
                    "tags": ["Admin"],
                    "security": [],
                    "responses": { "200": { "description": "This document" } }
                }
            }
        }
    })
}
