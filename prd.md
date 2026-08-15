# Face Recognition Authentication — Product Requirements Document

## 1. Document Status

- **Status:** Approved product direction
- **Project type:** Educational pet project / local MVP
- **Primary language:** English for source code, identifiers, API fields, documentation, UI copy, and notebook content
- **Deployment target:** Local development environment only

## 2. Product Summary

The project is a local educational prototype of face-based user authentication. A user creates a profile with a unique email address, first name, last name, and three face photographs. The system converts the photographs into face embeddings and stores only the resulting biometric template. The original photographs are not persisted.

During login, the user provides the registered email address and a new face photograph. The system performs one-to-one face verification against that user's stored template. A successful verification produces an access token and opens the user's protected profile page.

The product is explicitly an educational prototype. It must not be described or used as banking-grade, production-grade, or otherwise secure biometric authentication.

## 3. Goals

1. Build and fine-tune a face-embedding model on a public face dataset.
2. Learn and demonstrate the complete face-verification pipeline:
   - face detection;
   - landmark detection and alignment;
   - embedding generation;
   - biometric enrollment;
   - similarity-based verification;
   - validation-based threshold calibration;
   - API and UI integration.
3. Provide a functional local UI for profile registration, face login, protected profile access, and profile deletion.
4. Track training experiments and artifacts with MLflow.
5. Evaluate the model on identity-disjoint data rather than relying only on training loss or hand-picked examples.

## 4. Non-Goals

The MVP does not include:

- production or banking-grade security claims;
- liveness detection or presentation-attack detection;
- webcam or video input;
- password authentication;
- email ownership confirmation;
- account recovery;
- refresh tokens;
- login attempt limits, cooldowns, or rate limiting;
- an administrator panel or a list of all users;
- cloud deployment;
- Docker support;
- a mobile application;
- encryption of embeddings at rest;
- model-version migrations or support for a future `v2` model;
- HEIC, GIF, video, or other input formats outside JPEG and PNG.

## 5. Users and Primary Use Case

### 5.1 Intended User

A local user who wants to register a face profile and demonstrate biometric verification through a learning-oriented application.

### 5.2 Authentication Mode

The system uses **one-to-one verification**, not one-to-many identification:

1. The user claims an identity by entering a unique email address.
2. The system loads only the biometric template associated with that email.
3. The submitted face is compared only with that template.
4. The system either verifies or rejects the claim.

The system must not search for the closest person across all registered profiles.

## 6. User Flows

### 6.1 Profile Enrollment

1. The user opens the `Register` page.
2. The user enters:
   - email;
   - first name;
   - last name.
3. The user uploads exactly three face photographs taken with different angles.
4. Each image is validated and processed independently.
5. Each image must contain exactly one acceptable face.
6. The three generated embeddings must be mutually consistent enough to represent the same person.
7. The embeddings are averaged and L2-normalized to produce one enrollment centroid.
8. The profile and centroid are stored atomically.
9. The uploaded images are discarded after processing and are not persisted.

If the email already exists, enrollment fails. The MVP does not overwrite an existing biometric template.

### 6.2 Face Login

1. The user opens the `Login` page.
2. The user enters the registered email address.
3. The user uploads one new face photograph.
4. The system validates, detects, aligns, and embeds the face.
5. The system compares the login embedding with the stored enrollment centroid using cosine similarity.
6. The fixed verification threshold determines the result.
7. On success, the API issues an access token valid for two days.
8. On failure, the UI returns a generic verification error.

The user-facing response must not expose the similarity score. The MVP has no retry-limit or cooldown logic.

### 6.3 View Profile

After successful authentication, the user can open a protected `Profile` page that displays the profile's first name, last name, and email.

### 6.4 Delete Profile

An authenticated user can delete the current profile. Deletion must remove both the profile record and its stored face embedding data. The raw enrollment images do not exist and therefore require no cleanup.

## 7. Functional Requirements

### 7.1 Image Input

- Supported formats: JPEG and PNG only.
- Maximum size: 10 MB per file.
- The implementation must validate actual decodable image content rather than trusting only the filename extension or MIME type.
- EXIF orientation must be applied before face processing.
- Corrupted or unsupported images must be rejected.
- Enrollment and login images must contain exactly one face.
- Images with no face, multiple faces, an excessively small face, or excessive blur must be rejected with a clear machine-readable error.

Expected error categories include:

- `face_not_found`;
- `multiple_faces`;
- `face_too_small`;
- `image_too_blurry`;
- invalid or unsupported image errors;
- `profile_already_exists`;
- a generic verification failure.

The exact error schema and HTTP status mapping will be finalized during API design.

### 7.2 Profile Rules

- Email is the unique account identifier.
- First name and last name are profile attributes and are not identifiers.
- The MVP does not verify ownership of the provided email address.
- Existing profiles cannot be overwritten through enrollment.
- Updating a profile or biometric template is outside the MVP.
- Re-enrollment requires deleting the existing profile and creating it again.

### 7.3 Access Token

- Successful face verification returns an access token.
- The token lifetime is two days.
- Protected profile read and deletion operations require the token.
- Refresh tokens are not supported.
- Token secrets must come from local configuration or environment variables and must not be committed to Git.

## 8. ML System Design

### 8.1 Inference Pipeline

```text
JPEG/PNG image
    -> image validation and orientation correction
    -> YuNet face detection and five facial landmarks
    -> face alignment and 112 x 112 crop
    -> fine-tuned ResNet18 embedding model
    -> L2-normalized face embedding
    -> cosine similarity against enrollment centroid
    -> verified or rejected
```

### 8.2 Why Two Models Are Required

The application uses two inference models because detection and recognition are different tasks:

- **YuNet** locates a face and returns a bounding box, detection confidence, and five landmarks used for alignment.
- **ResNet18** converts the aligned face crop into an identity-relevant embedding.

YuNet cannot determine identity. ResNet18 should not receive an uncontrolled full image with arbitrary background, scale, and face orientation. The two models therefore form consecutive stages of one face-verification pipeline.

The ArcFace head is required only during training. It is not a third inference model and is not loaded by the API.

### 8.3 Detector and Alignment

- Use a pretrained OpenCV YuNet detector.
- Do not train a custom face detector.
- Use the returned landmarks to geometrically align the face.
- Produce a normalized `112 x 112` crop for the embedding model.
- Detection runs on CPU so it does not compete with ResNet18 training or inference for limited GPU memory.

### 8.4 Embedding Model

- Backbone: ImageNet-pretrained ResNet18.
- Training objective: ArcFace loss.
- Output: L2-normalized embedding.
- Input size: `112 x 112`.
- Exact embedding dimension is an experiment parameter to be selected and recorded before final training.
- The ArcFace classification head is discarded for application inference.

### 8.5 Enrollment Template

For each enrollment request:

1. Generate one normalized embedding for each of the three images.
2. Verify that the three embeddings are mutually consistent.
3. Reject the entire enrollment if the consistency requirement fails.
4. Average the three embeddings.
5. L2-normalize the average to create the stored centroid.

Enrollment is atomic: no partial profile or partial embedding data may remain after a failed request.

### 8.6 Verification

- Generate one normalized embedding from the login image.
- Compute cosine similarity with the stored enrollment centroid.
- Compare the similarity with a fixed threshold calibrated on validation data.
- Return only the verification result to the user-facing application.
- Similarity scores may be used in offline evaluation and MLflow artifacts, but not exposed by the login UI or public authentication response.

## 9. Dataset and Data Splitting

### 9.1 Training Dataset

- Use CelebA for non-commercial educational research.
- CelebA must not be committed to the repository.
- Include only identities with at least five images.
- Use a deterministic random seed for all dataset selection and splitting.

### 9.2 Identity-Disjoint Split

Split identities, not individual images:

- training identities: 80%;
- validation identities: 10%;
- test identities: 10%.

No person may appear in more than one split. This prevents identity leakage and measures generalization to people unseen during training.

### 9.3 Training Modes

- Full training uses the eligible CelebA data.
- The pipeline must support limiting identities and batches for quick development runs.
- The target training budget is approximately one to three hours on the local NVIDIA GTX 1650 with 4 GB VRAM.
- Google Colab may be used as an optional execution environment, but the local Python pipeline remains canonical.

### 9.4 External Evaluation

Use Labeled Faces in the Wild (LFW) verification pairs as an additional external evaluation. LFW is not the primary training dataset.

## 10. Training and Evaluation

### 10.1 Training Workflow

- The canonical workflow is a local Python training pipeline.
- Training must be launchable from the command line.
- The notebook may also launch training and conduct experiments.
- Shared implementation between `src/` and the notebook is preferred, but notebook experimentation is allowed and strict prohibition of duplicated experimental code is not a requirement.
- Use GPU acceleration and mixed precision where supported.
- Training configuration and random seeds must be recorded.

### 10.2 Notebook

Provide an English-language experiment notebook that can:

- inspect CelebA samples and identity distribution;
- demonstrate preprocessing and augmentation;
- run training experiments;
- visualize training and validation behavior;
- plot ROC curves and similarity distributions;
- inspect false acceptance and false rejection examples;
- load and evaluate a checkpoint.

### 10.3 Threshold Calibration

Threshold calibration must represent the real application scenario rather than only isolated image pairs:

1. Build three-image enrollment centroids for validation identities.
2. Create genuine login probes from the same identities.
3. Create impostor probes from different identities.
4. Calculate similarity distributions and an ROC curve.
5. Select a threshold targeting a validation `FAR <= 1%`.
6. Freeze the threshold before test evaluation.

The test set must not influence threshold selection.

### 10.4 Evaluation Metrics

Report at least:

- ROC AUC;
- Equal Error Rate (EER);
- False Acceptance Rate (FAR);
- False Rejection Rate (FRR);
- True Acceptance Rate at `FAR = 1%`;
- external LFW verification results.

Current learning targets are:

- `ROC AUC >= 0.90` on identity-disjoint test scenarios;
- `EER <= 15%`.

These values are evaluation targets, not claims of production security and not a blocker for creating the initial project skeleton. Results must be reported honestly without selecting a convenient test subset after evaluation.

### 10.5 Checkpoint Selection

- Calculate validation ROC AUC after each evaluation epoch.
- Select the checkpoint with the best validation ROC AUC.
- Export it to `checkpoints/best.pt` for application inference.
- Store the model state, preprocessing metadata, and calibrated threshold together or through an explicitly linked configuration.
- The test set must not determine the selected checkpoint.

The MVP assumes one fixed trained model. Model migration and compatibility with future checkpoints are outside scope.

## 11. Experiment Tracking with MLflow

MLflow runs locally and is used for both command-line and notebook experiments.

Recommended local layout:

- tracking metadata: local `mlflow.db`;
- artifacts: local `mlruns/`;
- experiment name: `face-recognition-arcface`.

Each relevant run should log:

- dataset and split configuration;
- random seed;
- backbone and embedding configuration;
- ArcFace parameters;
- optimizer and learning-rate configuration;
- batch size and epoch count;
- training and validation metrics by epoch;
- validation ROC AUC and threshold metrics;
- ROC and similarity-distribution figures;
- selected checkpoint and preprocessing metadata.

MLflow metadata, artifacts, and model checkpoints are local development artifacts and must not be committed to Git.

## 12. Application Architecture

### 12.1 Technology Stack

- Python
- PyTorch and torchvision
- OpenCV YuNet
- FastAPI backend
- Streamlit frontend
- SQLite persistence
- MLflow experiment tracking
- `uv` for dependency and environment management

### 12.2 Backend Responsibilities

The FastAPI backend owns:

- image validation;
- detection, alignment, and embedding inference;
- enrollment consistency validation;
- verification and threshold application;
- profile and embedding persistence;
- access-token creation and validation;
- profile deletion;
- API error contracts;
- model loading during application startup.

The API must fail at startup with a clear error when the required detector or trained embedding checkpoint is unavailable.

### 12.3 Frontend Responsibilities

The Streamlit UI is a client of the FastAPI backend. It must not implement independent ML inference or direct database access.

Required pages:

1. **Register**
   - email;
   - first name;
   - last name;
   - exactly three JPEG/PNG images.
2. **Login**
   - email;
   - one JPEG/PNG image.
3. **Profile**
   - protected display of the authenticated user's profile;
   - profile deletion action.

## 13. API Requirements

The minimum API surface is:

```text
POST   /profiles/enroll
POST   /auth/verify
GET    /profiles/me
DELETE /profiles/me
GET    /health
```

### 13.1 `POST /profiles/enroll`

Accepts profile fields and exactly three image uploads. Creates the profile and enrollment centroid only after all validation succeeds.

### 13.2 `POST /auth/verify`

Accepts an email and one image. Returns a two-day access token after successful verification. Returns a generic verification error on failure and does not expose similarity.

### 13.3 `GET /profiles/me`

Returns the authenticated user's profile information.

### 13.4 `DELETE /profiles/me`

Deletes the authenticated user's profile and all stored embedding data.

### 13.5 `GET /health`

Reports whether the API is running and whether required inference resources are loaded. It must not expose sensitive profile or model details.

## 14. Persistence

Use SQLite for the local MVP.

Conceptual records:

```text
Profile
- id
- email (unique)
- first_name
- last_name
- created_at

FaceTemplate
- profile_id
- enrollment_centroid
- created_at
```

The exact ORM and embedding serialization format will be selected during implementation. The database must never contain raw uploaded images.

## 15. Privacy and Security Boundaries

- Raw uploaded photographs must not be written to the database, project directories, logs, MLflow, or other application artifacts.
- Image bytes may exist transiently in process memory or request buffering while a request is processed.
- The application discards decoded images and request data after embedding generation.
- Embeddings are sensitive biometric information even though they are not photographs.
- Deleting a profile must also delete its biometric template.
- Logs must not contain image bytes, embeddings, access tokens, or unnecessary personal information.
- The absence of liveness detection means a photograph or screen replay may spoof the system.
- The absence of email confirmation means the first user can claim any unregistered email address.
- These limitations must be documented prominently in the README.

## 16. Repository and Artifact Policy

The repository contains source code, configuration templates, documentation, tests, and notebooks. It does not contain training data, trained checkpoints, runtime databases, or experiment artifacts.

At minimum, `.gitignore` must exclude:

- CelebA data and derived dataset files;
- `checkpoints/`;
- `mlruns/`;
- `mlflow.db`;
- local SQLite application databases;
- temporary uploads, caches, and generated training artifacts.

The README must explain:

- environment setup with `uv`;
- CelebA acquisition and expected local directory layout;
- YuNet model preparation;
- smoke training and full training commands;
- MLflow UI startup;
- API startup;
- Streamlit UI startup;
- model evaluation;
- local-only and non-production limitations.

## 17. Testing Requirements

Automated tests will be added in the final project phase.

Required automated coverage:

1. **API tests** for the agreed HTTP contract, validation, authentication, and error behavior.
2. **Integration tests** for the end-to-end application flow:
   - enroll a profile with three valid face inputs;
   - reject a duplicate email;
   - verify the enrolled user;
   - reject a different identity;
   - access the protected profile;
   - delete the profile and its embedding;
   - confirm that the deleted profile is no longer available.

Full CelebA training must not run as part of the normal automated test suite. The precise use of test doubles versus real checkpoints will be decided when the test phase is implemented.

## 18. MVP Acceptance Criteria

The MVP is functionally complete when:

1. The documented local environment can be installed with `uv`.
2. The CelebA preparation and identity-disjoint split are reproducible.
3. A local training run can produce an MLflow-tracked checkpoint.
4. Validation can calibrate and persist a verification threshold.
5. The evaluation pipeline reports all required metrics.
6. The API loads YuNet and the selected ResNet18 checkpoint.
7. The Streamlit UI supports registration, login, protected profile display, and deletion.
8. Raw uploaded images are not persisted by the application.
9. Profile deletion removes the stored embedding data.
10. Required API and integration tests pass.
11. The README clearly states all security and product limitations.

## 19. Implementation Sequence

1. Repository and `uv` project setup.
2. CelebA metadata preparation and identity-disjoint split.
3. Detection, alignment, and preprocessing pipeline.
4. ResNet18 embedding model and ArcFace training pipeline.
5. MLflow experiment tracking and notebook experiments.
6. Validation, threshold calibration, test evaluation, and checkpoint export.
7. SQLite persistence and profile domain logic.
8. FastAPI enrollment, verification, profile, deletion, and health endpoints.
9. Streamlit registration, login, and profile UI.
10. API and integration tests.
11. End-to-end local verification and final documentation.

## 20. Open Technical Decisions

The following implementation details were not fixed during product discovery and should be selected through experiments or focused technical design without changing the agreed product scope:

- embedding dimension;
- exact ArcFace scale and angular margin;
- optimizer, learning rate, scheduler, batch size, epoch count, and layer-freezing schedule;
- image augmentations;
- minimum face size and blur thresholds;
- enrollment-consistency threshold;
- exact validation/test pair generation counts;
- ORM and embedding serialization format;
- JWT implementation library and signing algorithm;
- exact API error schema and HTTP status codes;
- exact repository module layout;
- whether automated tests use deterministic inference doubles, real lightweight fixtures, or both.
