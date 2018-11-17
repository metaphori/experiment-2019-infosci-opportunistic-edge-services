package it.unibo.alchemist.loader.displacements

import it.unibo.alchemist.model.interfaces.Environment
import it.unibo.alchemist.model.interfaces.GeoPosition
import it.unibo.alchemist.model.interfaces.Position
import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.random.RandomGenerator
import org.apache.commons.math3.util.Pair
import java.util.stream.Stream

abstract class AbstractCloseTo<T, P: Position<P>> constructor(
    val randomGenerator: RandomGenerator,
    val environment: Environment<T, P>,
    val nodeCount: Int,
    val variance: Double
): Displacement<P> {

    init {
        require(nodeCount >= 0) { "The node count must be positive or zero: $nodeCount" }
        require(variance >= 0) { "The node count must be positive or zero: $nodeCount" }
    }

    var displacement: Collection<P>? = null

    protected fun covarianceMatrix(dimensions: Int): Array<out DoubleArray> = Array(dimensions) { index ->
        DoubleArray(dimensions) { if (it == index) variance else 0.0 }
    }

    protected abstract val sources: Sequence<DoubleArray>

    override final fun stream(): Stream<P> = (
        displacement
        ?: sources
            .map { MultivariateNormalDistribution(randomGenerator, it, covarianceMatrix(it.size)) }
            .map { Pair(1.0, it) }
            .let { it.toList() }
            .let { MixtureMultivariateNormalDistribution(randomGenerator, it) }
            .let { distribution ->
                (0 until nodeCount).map { environment.makePosition(*distribution.sample().toTypedArray()) }
            }.also { displacement = it }
        ).stream()
}