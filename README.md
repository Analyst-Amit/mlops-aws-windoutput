# MLOps Project

## Overview
This repository contains the template for an MLOps pipeline, including CI/CD workflows, model deployment, and testing.


### CI/CD Workflow

- **CI Pipeline**: Triggered on pull requests to the `main` branch. Runs unit tests and integration tests in a staging environment.
- **Release Pipeline**: Triggered on tag pushes (e.g., `v1.0`). Deploys jobs to production and creates a GitHub release.
