---
fed-number: 0003
title: FED Template
authors: ["@LorenzoMinto", "@danieljanes"]
creation-date: 2023-10-01
last-updated: 2023-10-01
status: provisional
---

# FED Template

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
- [Drawbacks](#drawbacks)
- [Alternatives Considered](#alternatives-considered)
- [Appendix](#appendix)

## Summary

In the rest communication model, a token-based system is necessary in order to limit users' contributions to only those that were explicitly requested by the server, i.e. users cannot submit results for a task they haven't received. This should help prevent basic poisoning attacks. However, a simple token-based system creates a 1-1 correspondence between request and response, allowing a potentially malicious server to craft a special task to extract some particular information from the requesting user. Hence the question: how can we authenticate users' contributions while maintaining the anonymity and unlinkability of the contributions?

[TODO - sentence 2: summary of the solution]
We propose using blinded signatures to achieve this, similar to their use in privacy pass protocol (https://privacypass.github.io/) and e-cash. The blinded signature makes it possible to break the link between task *issuance* and result *redemption* which is our goal.


## Motivation

In the rest communication model, a token-based system is necessary in order to limit users' contributions to only those that were explicitly requested by the server, i.e. users cannot submit results for a task they haven't received. This should help prevent basic poisoning attacks. However, a simple token-based system creates a 1-1 correspondence between request and response, allowing a potentially malicious server to craft a special task to extract some particular information from the requesting user. Hence the question: how can we authenticate users' contributions while maintaining the anonymity and unlinkability of the contributions?

### Goals

Guarantee the unlinkability between task issued and tasks results (contributions) submitted, essentially voiding the possibility of singling out a particular user for identification.

### Non-Goals

[TODO]

## Proposal

We propose using blinded signatures to achieve this, similar to their use in privacy pass protocol (https://privacypass.github.io/) and e-cash (http://www.hit.bme.hu/~buttyan/courses/BMEVIHIM219/2009/Chaum.BlindSigForPayment.1982.PDF). The blinded signature makes it possible to break the link between task *issuance* and result *redemption* which is our goal.

The protocol would look like the following:
- client (provider) chooses token x at random and hashes it into an eliptic curve, forms c(x) (blinded token) and supplies c(x) to server (signer)
- server (signer) signs c(x) by applying s' and returns the signed blinded token s'(c(x)) to client (provider)
- client (provider) strips signed blinded token by application of c', yielding c'(s'(c(x))) = s'(x)
- anyone can check that the blinded token s'(x) was formed by the signer by applying the signer's public key s and checking that s(s'(x)) is on the eliptic curve.

## Drawbacks

Using blinded tokens entails that althought the server will still be able to verify that the redeemed token for the contribution is a valid one, it will not be able to associate it with a particular task or even a particular set of tasks. To address this issue, the server could use different signatures depending on the task "batch id". What would the utility to the server be of being able to link the task result redeemed to a particular batch of tasks?

## Alternatives Considered

### [Alternative 1]

[TODO]

## Appendix
