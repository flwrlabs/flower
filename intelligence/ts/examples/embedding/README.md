---
tags: [node, minimal, typescript, embedding]
---

# Embedding TypeScript example

## Project setup

You must first download the example with the following command:

```bash
git clone --depth=1 https://github.com/flwrlabs/flower.git _tmp && mv _tmp/intelligence/ts/examples/embedding . && rm -rf _tmp && cd embedding
```

You can then install the project dependencies with:

```bash
npm i
```

## Build

If you want to build the project, you can use:

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
