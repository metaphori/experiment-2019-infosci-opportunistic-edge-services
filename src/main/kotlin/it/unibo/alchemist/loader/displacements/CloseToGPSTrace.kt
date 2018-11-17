package it.unibo.alchemist.loader.displacements

import it.unibo.alchemist.boundary.gpsload.impl.TraceLoader
import it.unibo.alchemist.kotlin.plus
import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.Environment
import it.unibo.alchemist.model.interfaces.GeoPosition
import it.unibo.alchemist.model.interfaces.Position
import it.unibo.alchemist.model.interfaces.Time
import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution
import org.apache.commons.math3.distribution.MultivariateNormalDistribution
import org.apache.commons.math3.random.RandomGenerator
import org.apache.commons.math3.util.Pair
import java.util.stream.Stream

class CloseToGPSTrace<T> @JvmOverloads constructor(
    randomGenerator: RandomGenerator,
    environment: Environment<T, GeoPosition>,
    nodeCount: Int,
    variance: Double,
    val from: Time = DoubleTime.ZERO_TIME,
    val interval: Time = DoubleTime(1.0),
    val to: Time = DoubleTime.INFINITE_TIME,
    gpsFilePath: String,
    normalizerClass: String,
    vararg normalizerArguments: Any
): AbstractCloseTo<T, GeoPosition>(randomGenerator, environment, nodeCount, variance) {

    val traces = TraceLoader(gpsFilePath, normalizerClass, *normalizerArguments)
    override val sources = traces.asSequence()
        .flatMap { trace ->
            generateSequence(from) { it + interval }
                .takeWhile { it <= to }
                .map { trace.interpolate(it) }
                .map { doubleArrayOf(it.latitude, it.longitude) }
        }
}