import { startClientInternal, Context, Client, GetParametersIns, GetParametersRes, FitIns, FitRes, EvaluateIns, EvaluateRes } from "../src/lib";
import { Code } from "../src/lib/typing";

class CustomClient extends Client {
  getParameters(ins: GetParametersIns): GetParametersRes {
    return {} as GetParametersRes;
  }
  fit(ins: FitIns): FitRes {
    const bytes = new Uint8Array(4);
    bytes[0] = 256;
    return { parameters: { tensors: [bytes, bytes], tensorType: "numpy.ndarray" }, numExamples: 1, metrics: {}, status: { code: Code.OK, message: "OK" } } as FitRes;
  }
  evaluate(ins: EvaluateIns): EvaluateRes {
    return {} as EvaluateRes;
  }
}

async function main() {
  const clientfn = (context: Context) => new CustomClient(context);
  const address = "127.0.0.1:9092";

  try {
    await startClientInternal(address, {}, undefined, null, clientfn, true, null, null, null, null, null);
  } catch (error) {
    console.error(`Failed to start client: ${error}`);
  }
}

main().then();
