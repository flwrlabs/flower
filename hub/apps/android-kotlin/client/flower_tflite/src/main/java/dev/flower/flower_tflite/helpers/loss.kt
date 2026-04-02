package dev.flower.flower_tflite.helpers

import dev.flower.flower_tflite.Sample
import kotlin.math.ln

fun <X> negativeLogLikelihoodLoss(
    samples: MutableList<Sample<X, FloatArray>>,
    logits: Array<FloatArray>
): Float = averageLossWith(samples, logits) { sample, logit ->
    -ln(logit[sample.label.argmax()])
}

fun <X, Y> averageLossWith(
    samples: MutableList<Sample<X, Y>>,
    logits: Array<Y>,
    loss: (Sample<X, Y>, logit: Y) -> Float
): Float {
    var lossSum = 0f
    for ((sample, logit) in samples lazyZip logits) {
        lossSum += loss(sample, logit)
    }
    return lossSum / samples.size
}
