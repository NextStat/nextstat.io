//! Auto-generated OpenAPI 3.1 specification for the NextStat server.
//!
//! Served at `GET /v1/openapi.json`.

use serde_json::{Value, json};

pub fn openapi_spec() -> Value {
    let server_tool_schema_example: Value = serde_json::from_str(include_str!(
        "../../../docs/specs/nextstat_tool_schema_server_v1.example.json"
    ))
    .expect("server tool schema example must parse");

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
                    "description": "Provide either `workspace` (full workspace JSON) or `model_id` (cached model key).",
                    "anyOf": [
                        { "required": ["workspace"] },
                        { "required": ["model_id"] }
                    ],
                    "properties": {
                        "workspace": { "type": "object", "description": "pyhf, HS3, or simplified-likelihood workspace JSON" },
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
                },
                "RateLimitError": {
                    "type": "object",
                    "required": ["error", "retry_after_s"],
                    "properties": {
                        "error": { "type": "string" },
                        "retry_after_s": { "type": "integer" }
                    }
                },
                "ToolDefinition": {
                    "type": "object",
                    "required": ["type", "function"],
                    "properties": {
                        "type": { "const": "function" },
                        "function": {
                            "type": "object",
                            "required": ["name", "description", "parameters"],
                            "properties": {
                                "name": { "type": "string" },
                                "description": { "type": "string" },
                                "parameters": { "type": "object" }
                            },
                            "additionalProperties": true
                        }
                    },
                    "additionalProperties": false
                },
                "ToolPolicy": {
                    "type": "object",
                    "required": ["availability", "reason_code", "reason"],
                    "properties": {
                        "availability": { "type": "string", "enum": ["exposed", "local_only"] },
                        "reason_code": { "type": "string" },
                        "reason": { "type": "string" }
                    }
                },
                "ToolCapability": {
                    "type": "object",
                    "required": ["name", "local_available", "server_available", "server_policy"],
                    "properties": {
                        "name": { "type": "string" },
                        "local_available": { "type": "boolean" },
                        "server_available": { "type": "boolean" },
                        "server_policy": { "$ref": "#/components/schemas/ToolPolicy" }
                    }
                },
                "ToolGuidanceRecipe": {
                    "type": "object",
                    "required": ["id", "transport", "title", "summary", "prompt", "tools", "docs"],
                    "properties": {
                        "id": { "type": "string" },
                        "transport": { "type": "string", "enum": ["server"] },
                        "title": { "type": "string" },
                        "summary": { "type": "string" },
                        "prompt": { "type": "string" },
                        "tools": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "docs": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    }
                },
                "ToolGuidance": {
                    "type": "object",
                    "required": ["hints", "recipes"],
                    "properties": {
                        "hints": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "recipes": {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/ToolGuidanceRecipe" }
                        }
                    }
                },
                "ToolSchemaResponse": {
                    "type": "object",
                    "required": ["schema_version", "transport", "tools", "capabilities", "guidance"],
                    "properties": {
                        "schema_version": { "const": "nextstat.tool_schema.v1" },
                        "transport": { "type": "string", "enum": ["server"] },
                        "tools": {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/ToolDefinition" }
                        },
                        "capabilities": {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/ToolCapability" }
                        },
                        "guidance": {
                            "$ref": "#/components/schemas/ToolGuidance"
                        }
                    }
                },
                "ToolExecuteRequest": {
                    "type": "object",
                    "required": ["name", "arguments"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Server-safe tool name returned by GET /v1/tools/schema."
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments object matching the selected tool's JSON schema.",
                            "additionalProperties": true
                        }
                    }
                },
                "ToolError": {
                    "type": "object",
                    "required": ["type", "message"],
                    "properties": {
                        "type": { "type": "string" },
                        "message": { "type": "string" }
                    },
                    "additionalProperties": true
                },
                "ToolMeta": {
                    "type": "object",
                    "required": [
                        "tool_name",
                        "nextstat_version",
                        "deterministic",
                        "eval_mode",
                        "threads_requested"
                    ],
                    "properties": {
                        "tool_name": { "type": "string" },
                        "nextstat_version": { "type": ["string", "null"] },
                        "deterministic": { "type": "boolean" },
                        "eval_mode": { "type": "string" },
                        "threads_requested": { "type": ["integer", "null"] },
                        "threads_applied": { "type": ["integer", "null"] },
                        "device": { "type": ["string", "null"] },
                        "warnings": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    },
                    "additionalProperties": true
                },
                "ToolResultEnvelope": {
                    "type": "object",
                    "description": "Stable tool result envelope. Tool-level failures still return HTTP 200 with ok=false inside this payload.",
                    "required": ["schema_version", "ok", "result", "error", "meta"],
                    "properties": {
                        "schema_version": { "const": "nextstat.tool_result.v1" },
                        "ok": { "type": "boolean" },
                        "result": {},
                        "error": {
                            "anyOf": [
                                { "type": "null" },
                                { "$ref": "#/components/schemas/ToolError" }
                            ]
                        },
                        "meta": { "$ref": "#/components/schemas/ToolMeta" }
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
                    "responses": {
                        "200": {
                            "description": "JSON schema for all supported tools plus capability/policy metadata",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ToolSchemaResponse" },
                                    "example": server_tool_schema_example
                                }
                            }
                        },
                        "401": {
                            "description": "Unauthorized",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/Error" }
                                }
                            }
                        },
                        "429": {
                            "description": "Rate limited",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/RateLimitError" }
                                }
                            }
                        }
                    }
                }
            },
            "/v1/tools/execute": {
                "post": {
                    "summary": "Execute a tool (OpenAI function-calling)",
                    "tags": ["Tools"],
                    "description": "Executes one server-safe tool. Tool-level validation and domain failures return HTTP 200 with ok=false in the response envelope; auth and rate-limit failures use HTTP errors.",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/ToolExecuteRequest" }
                            }
                        }
                    },
                    "responses": {
                        "400": {
                            "description": "Malformed JSON body or invalid request shape",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/Error" }
                                }
                            }
                        },
                        "415": {
                            "description": "Unsupported media type; request body must use Content-Type: application/json",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/Error" }
                                }
                            }
                        },
                        "200": {
                            "description": "Tool execution result envelope",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/ToolResultEnvelope" }
                                }
                            }
                        },
                        "401": {
                            "description": "Unauthorized",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/Error" }
                                }
                            }
                        },
                        "429": {
                            "description": "Rate limited",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/RateLimitError" }
                                }
                            }
                        }
                    }
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
                    "responses": {
                        "200": { "description": "This document" },
                        "401": {
                            "description": "Unauthorized",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/Error" }
                                }
                            }
                        },
                        "429": {
                            "description": "Rate limited",
                            "content": {
                                "application/json": {
                                    "schema": { "$ref": "#/components/schemas/RateLimitError" }
                                }
                            }
                        }
                    }
                }
            }
        }
    })
}
