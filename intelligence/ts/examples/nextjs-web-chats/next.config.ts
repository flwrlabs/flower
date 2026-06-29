import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  turbopack: {
    root: process.cwd(),
  },
  serverExternalPackages: ['@flwr/flwr', '@huggingface/transformers', 'onnxruntime-node'],
};

export default nextConfig;
