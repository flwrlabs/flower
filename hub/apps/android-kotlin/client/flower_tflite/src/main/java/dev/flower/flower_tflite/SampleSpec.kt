package dev.flower.flower_tflite

/**
 * Specification of the samples for [FlowerClient].
 * [convertX] and [convertY] are needed only because of the limitation of generics,
 * so simply fill in `{ it.toTypedArray() }`.
 * Feel free to choose the loss and accuracy functions from [dev.flower.flower_tflite.helpers].
 * @param emptyY Create an array of empty [Y].
 * @param loss Given test samples and logits, calculate test loss.
 * @param accuracy Given test samples and logits, calculate test accuracy.
 * If "accuracy" does not make sense here, return `-1f`.
 */
data class SampleSpec<X, Y>(
    val convertX: (List<X>) -> Array<X>,
    val convertY: (List<Y>) -> Array<Y>,
    val emptyY: (Int) -> Array<Y>,
    val loss: (MutableList<Sample<X, Y>>, Array<Y>) -> Float,
    val accuracy: (MutableList<Sample<X, Y>>, Array<Y>) -> Float,
)
