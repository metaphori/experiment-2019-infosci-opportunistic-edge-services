package it.unibo.alchemist.model.implementations.timedistributions

import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.Environment
import it.unibo.alchemist.model.interfaces.Incarnation
import it.unibo.alchemist.model.interfaces.Molecule
import it.unibo.alchemist.model.interfaces.Node
import it.unibo.alchemist.model.interfaces.Time
import it.unibo.alchemist.model.interfaces.TimeDistribution
import java.lang.IllegalStateException

/**
 * This class models a distribution that follows the packet arrival times as described in
 * [EdgeCloudSim](https://ieeexplore.ieee.org/document/7946405).
 * The delay produced depends on a constant propagation delay plus the packet size divided by the bandwidth.
 *
 * @param T the type of the environment. In case propagationDela, packetSize or bandwidth are not parsable to [Double],
 * then they will
 * be used as identifiers for properties to be obtained through [the incarnation][Incarnation]
 *
 * @param propagationDelayMolecule The propagation delay molecule. If the string is parsable as [Double], then it will get used
 * as a
 * constant delay. Otherwise, the String will be used within [Incarnation.getProperty]
 *
 */
class SimpleNetworkArrivals<T> private constructor(
        val incarnation: Incarnation<T, *>,
        val environment: Environment<T, *>,
        val node: Node<T>,
        private val constantPropagationDelay: Double? = null,
        private val propagationDelayMolecule: Molecule? = null,
        private val propagationDelayProperty: String? = null,
        private val constantPacketSize: Double? = null,
        private val packetSizeMolecule: Molecule? = null,
        private val packetSizeProperty: String? = null,
        private val constantBandwidth: Double? = null,
        private val bandwidthMolecule: Molecule? = null,
        private val bandwidthProperty: String? = null,
        private val accessPointIdentificator: Molecule? = null,
        startTime: Time = DoubleTime.ZERO_TIME
) : TimeDistribution<T> {

    @JvmOverloads
    constructor(incarnation: Incarnation<T, *>,
        node: Node<T>,
        environment: Environment<T, *>,
        propagationDelay: Double,
        packetSize: Double,
        bandwidth: Double,
        accessPointIdentificator: Molecule? = null
    ): this(incarnation, environment, node,
            constantPropagationDelay = propagationDelay,
            constantPacketSize = packetSize,
            constantBandwidth = bandwidth,
            accessPointIdentificator = accessPointIdentificator)

    @JvmOverloads
    constructor(incarnation: Incarnation<T, *>,
        environment: Environment<T, *>,
        node: Node<T>,
        propagationDelay: Double,
        packetSize: Double,
        bandwidthMolecule: Molecule,
        bandwidthProperty: String,
        accessPointIdentificator: Molecule? = null
    ): this(incarnation, environment, node,
        constantPropagationDelay = propagationDelay,
        constantPacketSize = packetSize,
        bandwidthMolecule = bandwidthMolecule,
        bandwidthProperty = bandwidthProperty,
        accessPointIdentificator = accessPointIdentificator)

    @JvmOverloads
    constructor(incarnation: Incarnation<T, *>,
        environment: Environment<T, *>,
        node: Node<T>,
        propagationDelay: Double,
        packetSizeMolecule: Molecule,
        packetSizeProperty: String,
        bandwidthMolecule: Molecule,
        bandwidthProperty: String,
        accessPointIdentificator: Molecule? = null
    ): this(incarnation, environment, node,
        constantPropagationDelay = propagationDelay,
        packetSizeMolecule = packetSizeMolecule,
        packetSizeProperty = packetSizeProperty,
        bandwidthMolecule = bandwidthMolecule,
        bandwidthProperty = bandwidthProperty,
        accessPointIdentificator = accessPointIdentificator)

    @JvmOverloads
    constructor(incarnation: Incarnation<T, *>,
        environment: Environment<T, *>,
        node: Node<T>,
        propagationDelayMolecule: Molecule,
        propagationDelayProperty: String,
        packetSizeMolecule: Molecule,
        packetSizeProperty: String,
        bandwidthMolecule: Molecule,
        bandwidthProperty: String,
        accessPointIdentificator: Molecule? = null
    ): this(incarnation, environment, node,
        constantPropagationDelay = null,
        propagationDelayMolecule = propagationDelayMolecule,
        propagationDelayProperty = propagationDelayProperty,
        packetSizeMolecule = packetSizeMolecule,
        packetSizeProperty = packetSizeProperty,
        bandwidthMolecule = bandwidthMolecule,
        bandwidthProperty = bandwidthProperty,
        accessPointIdentificator = accessPointIdentificator)

    private var time = startTime
    private var genTime = time

    private val myNeighborhood
        get() = node.neighborhood

    private val Node<T>.isAccessPoint
        get() = accessPointIdentificator?.let { this.contains(it) } ?: false

    val Node<T>.neighborhood
        get() = environment.getNeighborhood(this).neighbors

    val bandwidth: Double
        get() = (constantBandwidth ?: incarnation.getProperty(node, bandwidthMolecule, bandwidthProperty)).let { bw ->
            accessPointIdentificator?.let { id ->
                if (node.isAccessPoint || myNeighborhood.isEmpty()) {
                    bw / Math.max(myNeighborhood.size, 1)
                } else {
                    val accesspoints = myNeighborhood.filter { it.isAccessPoint }
                    when (accesspoints.size) {
                        0 -> bw
                        1 -> bw / accesspoints.first().neighborhood.size
                        else -> throw IllegalStateException("node ${node.id} is connected to multiple access points: $accesspoints")
                    }
                }
            } ?: bw
        }

    val packetSize: Double
        get() = constantPacketSize ?:
                incarnation.getProperty(node, packetSizeMolecule, packetSizeProperty).takeIf {
                    it.isFinite() && it >= 0
                } ?: 1.0

    val propagationDelay: Double
        get() = constantPropagationDelay ?: incarnation.getProperty(node, propagationDelayMolecule, propagationDelayProperty)

    override fun update(curTime: Time, executed: Boolean, rate: Double, env: Environment<T, *>?) {
        // TODO: once the infrastructure of events and time distributions has been refactored, this must be rethought of
        /*
         * executed e rate = 0 -> eseguita
         *
         */
        if (rate == 0.0) {
            time = DoubleTime.INFINITE_TIME
        } else if (time.isInfinite) {
            time = curTime + propagationDelay + packetSize / bandwidth
        }
//        if (executed) {
//            time = curTime + propagationDelay + packetSize / bandwidth
//            genTime = time
//        } else {
//            if (rate > 0) {
//                time = maxOf(curTime, genTime)
//            } else {
//                time = DoubleTime.INFINITE_TIME
//            }
//        }
    }

    override fun getNextOccurence() = time

    override fun clone(currentTime: Time?) = TODO("See Alchemist issue #79")

    override fun getRate(): Double = 1 / (propagationDelay + packetSize / bandwidth)

}

private operator fun Time.plus(other: Double) = sum(DoubleTime(other))

