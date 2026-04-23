---
ep-number: 0001
title: Flower Enhancement Proposals
authors: [ "@nfnt" ]
creation-data: 2022-03-10
status: provisional
---

# Flower Enhancement Proposals

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
  - [Enhancement Proposal Template](#enhancement-proposal-template)
  - [Metadata](#metadata)
  - [Workflow](#workflow)
- [Drawbacks](#drawbacks)
- [Alternatives Considered](#alternatives-considered)
  - [Github Issues](#github-issues)
  - [Google Docs](#google-docs)

## Summary

An Enhancement Proposal is a standardized development process to
- provide a common structure for proposing larger changes
- ensure that the motivation for a change is clear
- persist project information in a version control system
- document the motivation for impactful user-facing changes
- reserve GitHub issues for tracking work in flight
- ensure community participants can successfully drive changes to completion across one or more releases while stakeholders are adequately represented throughout the process

Hence, an Enhancement Proposal combines aspects of
- a feature, and effort-tracking document
- a product requirements document
- a design document

into one file, which is created incrementally in collaboration with the community.

## Motivation

For far-fetching changes or features proposed to Flower, an abstraction beyond a single Github issues or pull request is required in order to understand and communicate upcoming changes to the project.

The purpose of this process is to reduce the amount of "tribal knowledge" in our community. By moving decisions from Slack threads, video calls and hallway conversations into a well tracked artifact, this process aims to enhance communication and discoverability.

### Goals

Roughly any larger, user facing enhancement should follow the Enhancement Proposal process. If an enhancement would be described in either written or verbal communication to anyone besides the author or developer, then consider creating an Enhancement Proposal.

Similarly, any technical effort (refactoring, major architectural change) that will impact a large section of the development community should also be communicated widely. The Enhancement Proposal process is suited for this even if it will have zero impact on the typical user or operator.

### Non-Goals

For small changes and additions, going through the Enhancement Proposal process would be time-consuming and unnecessary. This, e.g., includes adding new Federated Learning algorithms, as these only add features without changing how flower works or is used.

Enhancement Proposals are different from feature requests, as they are already providing a laid-out path for an implementation and are championed by members of the community.

## Proposal

An Enhancement Proposal consists of a Markdown file that follows a defined template and a workflow to review and store proposals for reference.

### Enhancement Proposal Template

Each proposal is provided as a Markdown file having the following structure

- Metadata  
  Metadata as [described below](#metadata) in form of a YAML preamble
- Table of Contents (if needed)
- Summary  
- Motivation
  - Goals
  - Non-Goals
- Proposal
  - Notes/Constraints/Caveats (optional)
- Design Details (optional)
  - Graduation Criteria
  - Upgrade/Downgrade Strategy (if applicable)
- Drawbacks
- Alternatives Considered

As reference, this document follows the above structure.

### Metadata

- __title__ (Required)  
  The title of the proposal in plain language.
- __status__ (Required)  
  The current status of the proposal. See [workflow](#workflow) for the possible states.
- __authors__ (Required)  
  A list of authors of the proposal. This is simply the Github ID.
- __creation-date__ (Required)  
  The date that the proposal was first submitted in a PR.
- __last-updated__ (Optional)
  The date that the proposal was last changed significantly.  
- __see-also__ (Optional)  
  A list of other proposals that are releavent to this one.
- __replaces__ (Optional)  
  A list of proposals that this one replaces.
- __superseded-by__ (Optional)  
  A list of proposals that this one supersedes.

### Workflow

The idea forming the enhancement should already have been discussed or pitched in the comminuty. As such, it needs a champion, usually the author, who shepherds the proposal. This person also has to find committers to flower willing to review the proposal.

New enhancement proposals are checked in with a file name in the form of `NNNN-YYYYMMDD-proposal-title.md`, with `NNNN` being the proposal number, to `proposals`. All proposals start in `provisional` state as part of a pull request. Discussions are done as part of the pull request review.

Once a proposal has been reviewed and approved, its status is changed to `implementable`. The actual implementation is then done in separate pull requests. These pull requests should mention the respective proposal as part of their description. After the implementation is done, the proposal status is changed to `implemented`.

Under certain conditions, other states are possible. An Enhancement Proposal has the following states:
- `provisional`: The Enhancement Proposal has been proposed and is actively being defined. This is the starting state while the proposal is being fleshed out and actively defined and discussed.
- `implementable`: The proposal has been reviewed and approved.
- `implemented`: The proposal has been implemented and is no longer actively changed.
- `deferred`: The proposal is proposed but not actively being worked on.
- `rejected`: The authors and reviewers have decided that this proposal is not moving forward.
- `withdrawn`: The authors have withdrawn the proposal.
- `replaced`: The proposal has been replaced by a new proposal.

## Drawbacks

Adding an additional process to the ones already provided by Github (Issues and Pull Requests) adds more complexity and can be a barrier for potential first time contributors.

Expanding the proposal template beyond the single-sentence description currently required in the features issue template may be a heavy burden for non-native English speakers.

## Alternatives Considered

### Github Issues

Using Github Issues for these kinds of proposals is doable. By, e.g., using tags, one could differentiate and filter them from other issues. The main issue is in discussing and reviewing a proposal: Github issues only have a single thread for comments. Proposals usually have multiple thread of discussion at the same time for various parts of the proposal. Managing these multiple discussions can be confusing when using Github Issues.

### Google Docs

Google Docs allow for multiple threads of discussions. But as Google Docs are hosted outside the project, their discoverability by the community needs to be taken care of. A list of links to all proposals has to be managed and made available for the community. Compared to shipping proposals as part of Flower's repository, the potential for missing links is much higher.
