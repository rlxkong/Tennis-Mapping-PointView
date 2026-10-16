# PointView ESP

**Affordable tennis video analysis, built around the way players actually review a match.**

PointView ESP is an entrepreneurship project exploring a more approachable way for tennis players and coaches to learn from match video. The concept brings video review, tennis-specific stats, and searchable notes into one workflow.

> **Project status:** This repository is a project scaffold and product overview. No application source code, trained models, or working analysis pipeline was included when it was prepared. The capabilities below describe the project vision, not verified implementation.

## The idea

Players record a match, upload the footage, and review the moments that matter without spending hours scrubbing through inactive play. PointView ESP is intended to make that process easier to use and more affordable than traditional video-analysis options.

## Planned player workflow

1. Upload a tennis match video.
2. Trim or skip inactive stretches so review time focuses on play.
3. Explore tennis-relevant stats and video moments.
4. Add searchable notes to remember patterns, adjustments, and coaching feedback.

## Computer-vision direction

The project explored a computer-vision pipeline using:

- **YOLOv8 and BYTETrack** for player detection and tracking.
- **TrackNet** for tennis-ball tracking.
- **Court keypoints and homography** to map video positions to a court view.

These are proposed or explored technical directions; this repository does not currently contain models, weights, datasets, or a working pipeline. Any future implementation should document model versions, data sources, evaluation results, and applicable licenses.

## Product principles

- **Accessible:** Keep the experience and intended cost approachable for players.
- **Useful:** Connect statistics to video moments and notes that support practice.
- **Respectful of player data:** Explain how uploaded videos are stored, used, and deleted before accepting real match footage.
- **Honest about analysis:** Show uncertainty and avoid presenting estimates as official match statistics.

## Repository status

This repository currently contains project documentation only. There is no installable app, API, dataset, model, or demo yet. As implementation is added, update this section with the supported platforms, setup instructions, and a real demo link.

## Getting started

There is no application to install yet. For now, start with the project overview and roadmap:

- [Project roadmap](ROADMAP.md)

When a runnable implementation is added, replace this section with verified setup steps and sample inputs that can be shared publicly.

## Potential next steps

- Confirm the target user and first release scope with players and coaches.
- Choose a video-upload and storage approach, including retention and deletion behavior.
- Define a small, consented evaluation set and metrics for tracking quality.
- Build and evaluate one end-to-end analysis workflow before expanding features.
- Add setup instructions, screenshots, and a short demo after those pieces exist.

## Team

PointView ESP was co-founded by three tennis teammates. Add founder names and contact links here when the team is ready to publish them.

## License

No license has been selected yet. Until one is added, do not assume this project is available for reuse. Choose a license with the project team before inviting contributions or reuse.
