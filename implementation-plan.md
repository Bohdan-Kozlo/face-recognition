# Face Recognition Authentication — Implementation Plan

## 1. Purpose

This plan turns the approved [PRD](./prd.md) into an executable sequence of vertical milestones. Each milestone has a concrete outcome, dependencies, implementation tasks, and completion evidence.

The plan intentionally builds the ML pipeline before the application depends on it. API and integration tests are added in the final project phase, as agreed. Focused manual checks and short smoke commands are still required throughout development so that failures are found near the milestone that introduced them.

## 2. Delivery Strategy

### 2.1 Principles

- Keep the local Python pipeline canonical.
- Deliver one verifiable outcome per milestone.
- Separate training concerns from application inference concerns.
- Never use test identities to select a checkpoint or threshold.
- Never persist uploaded face images.
- Keep CelebA, checkpoints, databases, and MLflow artifacts outside Git.
- Record experiment decisions instead of relying on notebook state.
- Do not start UI work until the API contract and inference service are usable.

### 2.2 Milestone Dependency Map

```text
M0 Project bootstrap
 ├─> M1 Data preparation
 └─> M2 Face preprocessing
       └─> M3 Embedding model and ArcFace
             └─> M4 Training and MLflow
                   └─> M5 Evaluation and checkpoint export
                         └─> M6 Inference service
                               └─> M7 Persistence and authentication
                                     └─> M8 FastAPI application
                                           └─> M9 Streamlit UI
                                                 └─> M10 API and integration tests
                                                       └─> M11 Final verification and documentation
```

## 3. Milestone 0 — Project Bootstrap

### Outcome

A reproducible local Python project that has a stable directory structure, dependency groups, configuration entry points, and artifact boundaries.

### Tasks

1. Initialize Git if the repository is not already version-controlled.
2. Initialize a `uv` Python project.
3. Choose and document the supported Python version.
4. Add dependency groups for:
   - core ML and image processing;
   - training and experiment tracking;
   - API;
   - UI;
   - notebook;
   - development and testing.
5. Create the initial package and directory structure.
6. Add `.gitignore` rules for:
   - `data/` and CelebA files;
   - `checkpoints/`;
   - `mlruns/`;
   - `mlflow.db`;
   - local application databases;
   - temporary uploads and generated artifacts;
   - Python, notebook, IDE, and `uv` caches.
7. Add `.env.example` with non-secret configuration names.
8. Add a minimal English README with setup placeholders and the educational-security disclaimer.
9. Add a central typed settings module for paths and runtime configuration.
10. Add lightweight formatting and static-analysis commands.

### Proposed Structure

```text
face-recognition/
├── data/                              # CelebA and manifests; ignored
├── models/                            # YuNet ONNX; ignored
├── checkpoints/                       # best.pt; ignored
├── notebooks/
│   └── face_recognition.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── vision.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   ├── engine.py
│   ├── storage.py
│   ├── security.py
│   ├── api.py
│   └── ui.py
├── tests/
│   ├── test_api.py
│   └── test_integration.py
├── .env.example
├── .gitignore
├── implementation-plan.md
├── prd.md
├── pyproject.toml
├── README.md
└── uv.lock
```

The project intentionally uses flat sibling modules directly inside `src/`. It does not introduce a second package directory or separate subdirectories for API, training, persistence, and UI. A module should become a subdirectory only after its implementation genuinely needs multiple cohesive files.

`engine.py` is the main seam between the application and the face-recognition implementation. It exposes enrollment and verification while hiding image validation, YuNet detection, alignment, embedding generation, centroid construction, and threshold comparison. `storage.py` is the seam between application logic and SQLite. No additional adapter abstractions are required for the MVP.

### Completion Evidence

- A clean environment can be created with `uv sync`.
- The source modules import successfully.
- Configuration paths resolve from the repository root.
- Ignored runtime artifacts do not appear in `git status`.
- Formatting and static checks run successfully on the bootstrap code.

## 4. Milestone 1 — CelebA Data Preparation

### Dependencies

- Milestone 0

### Outcome

A deterministic, identity-disjoint CelebA manifest that can drive training, validation, and test workflows without leaking identities.

### Tasks

1. Document how to obtain CelebA under its non-commercial research terms.
2. Define the expected local dataset directory structure.
3. Parse CelebA identity annotations and image paths.
4. Validate that every referenced image exists and is decodable.
5. Calculate dataset statistics:
   - total images;
   - total identities;
   - images per identity;
   - excluded identities and reasons.
6. Remove identities with fewer than five valid images.
7. Split identities deterministically into `80/10/10` train, validation, and test groups.
8. Assert that the identity sets do not overlap.
9. Save generated manifests locally under ignored data directories.
10. Add CLI options for:
    - random seed;
    - input data root;
    - output manifest directory;
    - identity limit for development runs.
11. Produce a small summary artifact that can be logged to MLflow later.

### Decisions to Resolve

- Exact manifest format: CSV, Parquet, or JSON Lines.
- Whether to use all eligible identities immediately or stage the first experiments on a deterministic subset.

### Completion Evidence

- Re-running preparation with the same seed produces the same manifests.
- Every identity belongs to exactly one split.
- Every retained identity has at least five valid images.
- A manifest summary reports counts for all three splits.
- A limited development manifest can be generated quickly.

## 5. Milestone 2 — Face Detection, Alignment, and Image Validation

### Dependencies

- Milestone 0

### Outcome

A single preprocessing component that converts one valid uploaded or dataset image into an aligned `112 x 112` face crop, or returns a precise domain error.

### Tasks

1. Define the YuNet model asset path and local acquisition instructions.
2. Load YuNet through OpenCV on CPU.
3. Implement safe JPEG/PNG decoding from bytes.
4. Enforce the 10 MB input limit.
5. Apply EXIF orientation before detection.
6. Detect faces and parse bounding box, landmarks, and confidence.
7. Reject images with:
   - no face;
   - multiple faces;
   - a face below the minimum size;
   - excessive blur;
   - invalid or unsupported content.
8. Implement landmark-based geometric alignment.
9. Produce an RGB `112 x 112` crop in the exact value range expected by ResNet18.
10. Define preprocessing metadata that can be saved with a checkpoint.
11. Add a small manual inspection script or notebook section that renders accepted crops and rejected examples without persisting API uploads.

### Decisions to Resolve Experimentally

- YuNet confidence threshold.
- Minimum face dimensions or face-area ratio.
- Blur metric and threshold.
- Landmark template and alignment transform.
- Pixel normalization values used by the embedding model.

### Completion Evidence

- Valid single-face samples produce aligned `112 x 112` RGB crops.
- No-face and multi-face samples produce distinct errors.
- Corrupted input is rejected before model inference.
- Alignment output is visually inspected on different poses.
- The component accepts in-memory bytes and does not require saving an upload to disk.

## 6. Milestone 3 — ResNet18 Embedder and ArcFace Objective

### Dependencies

- Milestone 0
- Milestone 2 preprocessing contract

### Outcome

A trainable face-embedding model and ArcFace training head with a stable inference interface.

### Tasks

1. Create an ImageNet-pretrained ResNet18 backbone.
2. Replace the classification layer with an embedding projection.
3. L2-normalize embeddings in the model's inference output.
4. Implement the ArcFace classification head and loss path.
5. Keep ArcFace outside the exported inference model.
6. Define a checkpoint schema containing:
   - embedder state dictionary;
   - architecture name;
   - embedding dimension;
   - preprocessing metadata;
   - training configuration summary;
   - calibrated verification threshold when available.
7. Add device selection for CPU and CUDA.
8. Add mixed-precision compatibility for CUDA training.
9. Add a synthetic forward/backward smoke command.

### Decisions to Resolve Experimentally

- Embedding dimension, starting with either 128 or 256.
- ArcFace scale and angular margin.
- Whether early backbone stages begin frozen and are later unfrozen.

### Completion Evidence

- A synthetic batch produces normalized embeddings of the configured size.
- Embedding norms are approximately one.
- ArcFace produces finite logits and loss.
- A backward pass updates trainable parameters.
- The embedder can be saved and loaded without the ArcFace head.

## 7. Milestone 4 — Training Pipeline, Notebook, and MLflow

### Dependencies

- Milestone 1
- Milestone 2
- Milestone 3

### Outcome

A local training workflow that can run a short smoke experiment or a full CelebA fine-tuning run while recording reproducible experiments in MLflow.

### Tasks

1. Build the CelebA training dataset and data loaders from manifests.
2. Add training augmentations appropriate for aligned faces.
3. Implement the training loop with:
   - CUDA support;
   - automatic mixed precision;
   - optimizer and scheduler;
   - epoch and batch progress;
   - graceful interruption;
   - checkpoint recovery if included in the selected scope.
4. Add CLI configuration for:
   - data manifests;
   - embedding dimension;
   - ArcFace parameters;
   - batch size;
   - learning rate;
   - epoch count;
   - random seed;
   - identity and batch limits.
5. Configure local MLflow tracking:
   - metadata in `mlflow.db`;
   - artifacts in `mlruns/`;
   - experiment name `face-recognition-arcface`.
6. Log parameters, epoch metrics, configuration, and checkpoints.
7. Create the English experiment notebook.
8. Let the notebook inspect data, launch experiments, query MLflow results, and visualize training behavior.
9. Document the commands for smoke training, full training, and MLflow UI startup.

### Initial Experiment Sequence

1. Synthetic forward/backward smoke run.
2. One-batch overfit experiment.
3. Small deterministic identity subset.
4. Compare embedding dimensions if needed.
5. Tune batch size to remain within 4 GB VRAM.
6. Run the first full eligible CelebA experiment.

### Completion Evidence

- A smoke run completes in minutes.
- A run appears in the local MLflow UI with parameters and metrics.
- Training can resume from source configuration rather than notebook state alone.
- The GTX 1650 run does not exceed available VRAM at the selected batch size.
- The notebook can reproduce or inspect a recorded experiment.

## 8. Milestone 5 — Validation, Threshold Calibration, and Final Evaluation

### Dependencies

- Milestone 4 trained checkpoint

### Outcome

A selected checkpoint, a validation-calibrated threshold, and an honest evaluation report on unseen identities.

### Tasks

1. Build validation scenarios that match the product flow:
   - three images form an enrollment centroid;
   - another image from the same identity forms a genuine probe;
   - images from other identities form impostor probes.
2. Determine the enrollment-consistency threshold using validation identities.
3. Generate genuine and impostor cosine-similarity distributions.
4. Calculate:
   - ROC AUC;
   - EER;
   - FAR;
   - FRR;
   - `TAR@FAR=1%`.
5. Select the verification threshold targeting validation `FAR <= 1%`.
6. Select the best checkpoint using validation ROC AUC only.
7. Freeze the checkpoint and threshold.
8. Evaluate once on identity-disjoint test scenarios.
9. Evaluate on LFW verification pairs as an external benchmark.
10. Log metrics, plots, selected threshold, and failure examples to MLflow.
11. Export `checkpoints/best.pt` with inference metadata.

### Decisions to Resolve

- Number and sampling policy for genuine and impostor scenarios.
- Confidence intervals or repeated sampling strategy for reported metrics.
- How many false-acceptance and false-rejection examples to inspect manually.

### Completion Evidence

- Validation, test, and LFW metrics are stored separately.
- The test set did not select the checkpoint or threshold.
- `best.pt` loads in a clean inference process.
- The checkpoint contains or references all preprocessing and threshold settings.
- Results are recorded even if the learning targets are not reached.

## 9. Milestone 6 — Application Inference Service

### Dependencies

- Milestone 2 preprocessing
- Milestone 5 exported checkpoint

### Outcome

A reusable application service that turns in-memory image bytes into embeddings and performs enrollment and verification decisions.

### Tasks

1. Load YuNet and `best.pt` once at application startup.
2. Validate checkpoint metadata against the runtime preprocessing configuration.
3. Implement `embed(image_bytes)`.
4. Implement three-image enrollment:
   - preprocess each image;
   - create normalized embeddings;
   - validate mutual consistency;
   - create and normalize the centroid.
5. Implement verification against a stored centroid.
6. Keep raw similarity available only inside the application service for diagnostics.
7. Map preprocessing and verification failures to stable domain errors.
8. Ensure request images and decoded arrays are not persisted or logged.

### Completion Evidence

- A local script can create a centroid from three images.
- A new image from the same person can be verified.
- A different person can be rejected in a manual demonstration.
- Startup fails clearly if YuNet or `best.pt` is absent or incompatible.
- No raw image artifact is created by the service.

## 10. Milestone 7 — Persistence and Token Authentication

### Dependencies

- Milestone 0
- Milestone 6 service contract

### Outcome

SQLite-backed profile storage with atomic enrollment, unique email enforcement, protected profile access, token issuance, and complete profile deletion.

### Tasks

1. Select the ORM and migration approach.
2. Define `Profile` and `FaceTemplate` persistence models.
3. Choose a deterministic embedding serialization format.
4. Enforce unique normalized email addresses.
5. Implement an atomic profile-plus-template creation transaction.
6. Implement profile lookup by email and by authenticated profile ID.
7. Implement cascade deletion of profile and template.
8. Implement JWT creation and validation.
9. Set access-token expiry to two days.
10. Read signing secrets from environment configuration.
11. Ensure logs never contain embeddings, image bytes, or access tokens.

### Decisions to Resolve

- ORM and migration library.
- Email normalization policy.
- Embedding binary format and dtype.
- JWT signing algorithm and claims.

### Completion Evidence

- Duplicate email creation is rejected.
- Failed enrollment does not leave a partial profile.
- A valid token resolves the correct profile.
- An expired or invalid token is rejected.
- Deleting a profile removes its template.

## 11. Milestone 8 — FastAPI Application

### Dependencies

- Milestone 6
- Milestone 7

### Outcome

A local FastAPI backend implementing the complete agreed API contract.

### Tasks

1. Add application lifespan loading for YuNet, the embedder checkpoint, settings, and database resources.
2. Implement a consistent error envelope and HTTP status mapping.
3. Implement:
   - `POST /profiles/enroll`;
   - `POST /auth/verify`;
   - `GET /profiles/me`;
   - `DELETE /profiles/me`;
   - `GET /health`.
4. Accept multipart forms and `UploadFile` inputs without persisting uploads.
5. Enforce the file count, type, size, and content requirements.
6. Keep similarity scores out of public authentication responses.
7. Add bearer-token protection to profile read and delete operations.
8. Add OpenAPI descriptions and representative error responses.
9. Add safe application logging.
10. Document the local startup command and required environment variables.

### Completion Evidence

- The OpenAPI page shows all five endpoints and schemas.
- Manual requests can enroll, verify, read, and delete a profile.
- Invalid images return specific validation errors.
- Failed verification returns a generic public error.
- Health output reflects model readiness without leaking sensitive details.

## 12. Milestone 9 — Streamlit UI

### Dependencies

- Milestone 8

### Outcome

A local Streamlit client that completes the entire user flow through FastAPI and never accesses the database or ML models directly.

### Tasks

1. Create shared API-client code with timeout and error handling.
2. Build the `Register` page:
   - email;
   - first name;
   - last name;
   - exactly three upload controls;
   - clear validation and result messaging.
3. Build the `Login` page:
   - email;
   - one upload control;
   - generic verification failure messaging.
4. Store the two-day access token in Streamlit session state for the active browser session.
5. Build the protected `Profile` page.
6. Add profile deletion with explicit confirmation.
7. Clear session state after deletion or invalid authentication.
8. Add the educational-prototype and no-liveness warning.
9. Keep UI copy and source code in English.

### Completion Evidence

- A user can register through the UI.
- The same user can log in with a new photograph.
- A different face is rejected.
- The profile page is inaccessible without a valid token.
- Profile deletion removes access and returns the UI to an unauthenticated state.

## 13. Milestone 10 — API and Integration Tests

### Dependencies

- Milestone 8
- Milestone 9 user flow complete

### Outcome

The agreed automated API and integration coverage is implemented at the end of the project.

### Tasks

1. Decide whether tests use deterministic inference doubles, lightweight real fixtures, or both.
2. Isolate test configuration, database, secrets, and artifacts.
3. Add API tests for:
   - request schemas;
   - image count and size rules;
   - invalid content;
   - no face and multiple faces;
   - duplicate email;
   - verification failure privacy;
   - missing, invalid, and expired tokens;
   - health behavior.
4. Add integration tests for:
   - successful three-image enrollment;
   - duplicate enrollment rejection;
   - successful verification;
   - different-identity rejection;
   - protected profile access;
   - profile and embedding deletion;
   - inability to access a deleted profile.
5. Verify that tests do not run CelebA training.
6. Verify that test uploads do not appear as files or logged artifacts.
7. Add a single documented command for the required suite.

### Completion Evidence

- API tests pass from a clean environment.
- The complete integration flow passes.
- Tests are deterministic and do not require CelebA.
- The test database and artifacts are isolated from local development data.
- No raw test upload is persisted by the application.

## 14. Milestone 11 — Final Verification and Documentation

### Dependencies

- All previous milestones

### Outcome

Another developer can set up, train, evaluate, run, and verify the project locally using the repository documentation.

### Tasks

1. Complete the README with:
   - project overview;
   - architecture and two-model explanation;
   - security limitations;
   - `uv` setup;
   - CelebA setup;
   - YuNet setup;
   - data preparation;
   - smoke and full training;
   - MLflow UI;
   - evaluation;
   - FastAPI startup;
   - Streamlit startup;
   - required tests.
2. Add example environment configuration without secrets.
3. Verify every documented command in order.
4. Run the final API and integration suites.
5. Run a manual acceptance flow using a real checkpoint.
6. Confirm that no raw face images are present in runtime artifacts or logs.
7. Confirm that ignored data, databases, MLflow runs, and checkpoints are absent from Git status.
8. Record final evaluation results and known limitations.
9. Compare the delivered behavior against every PRD acceptance criterion.

### Final Completion Evidence

- Clean `uv` setup succeeds.
- Data preparation is reproducible.
- Training produces an MLflow-tracked checkpoint.
- Evaluation produces the required metrics and fixed threshold.
- API and UI complete the full user flow.
- Profile deletion removes the embedding.
- API and integration tests pass.
- Documentation is accurate and complete.
- The repository contains no dataset, checkpoint, runtime database, MLflow store, secrets, or raw uploaded face images.

## 15. Recommended Work Order

Execute milestones in this order:

1. **M0:** Project bootstrap
2. **M1:** CelebA preparation
3. **M2:** Face preprocessing
4. **M3:** ResNet18 and ArcFace
5. **M4:** Training, notebook, and MLflow
6. **M5:** Evaluation and checkpoint export
7. **M6:** Application inference service
8. **M7:** Persistence and JWT
9. **M8:** FastAPI
10. **M9:** Streamlit
11. **M10:** API and integration tests
12. **M11:** Final verification and documentation

Do not begin M6 against a placeholder checkpoint if M5 can be completed first. A thin fake inference seam may be introduced to unblock application structure, but the real end-to-end acceptance flow must use the exported checkpoint.

## 16. Progress Tracking Template

Use this checklist as the project-level tracker:

- [ ] M0 — Project bootstrap
- [ ] M1 — CelebA data preparation
- [ ] M2 — Face detection, alignment, and validation
- [ ] M3 — ResNet18 embedder and ArcFace objective
- [ ] M4 — Training pipeline, notebook, and MLflow
- [ ] M5 — Validation, threshold calibration, and final evaluation
- [ ] M6 — Application inference service
- [ ] M7 — Persistence and token authentication
- [ ] M8 — FastAPI application
- [ ] M9 — Streamlit UI
- [ ] M10 — API and integration tests
- [ ] M11 — Final verification and documentation

For each completed milestone, record:

- the commands that were run;
- the generated evidence or MLflow run ID;
- unresolved limitations;
- any PRD decision that changed and why.
