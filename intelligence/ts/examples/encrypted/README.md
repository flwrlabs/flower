---
tags: [node, minimal, remote, encryption, Flower Confidential Remote Compute, typescript]
---

# Flower Confidential Remote Compute example

## Project setup

You must first download the example with the following command:

```bash
git clone --depth=1 https://github.com/flwrlabs/flower.git _tmp && mv _tmp/intelligence/ts/examples/encrypted . && rm -rf _tmp && cd encrypted
```

You can then install the project dependencies with:

```bash
npm i
```

> [!WARNING]
> In order to run this example, you need to update `fi.apiKey = 'REPLACE_HERE'` inside `src/index.ts` with a valid API key (if you don't have one, you can register [here](https://flower.ai/intelligence/)).

## Build

Then, you need to build the project:

```bash
npm run build
```

## Run

In order to run the example once the project has been built:

```bash
node dist/index.js
```

or

```bash
npm run start
```
