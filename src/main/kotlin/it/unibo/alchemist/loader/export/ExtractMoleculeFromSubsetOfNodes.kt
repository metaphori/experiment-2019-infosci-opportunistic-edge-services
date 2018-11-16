package it.unibo.alchemist.loader.export

import it.unibo.alchemist.model.interfaces.*
import it.unibo.alchemist.model.interfaces.Time
import java.lang.IllegalStateException

class ExtractMoleculeFromSubsetOfNodes<T>(
    val incarnation: Incarnation<T, *>,
    val molecule: Molecule,
    val property: String,
    val fromId: Int,
    val numberOfNodes: Int
) : Extractor {
    override fun extractData(env: Environment<*, *>, r: Reaction<*>, time: Time, step: Long): DoubleArray =
        (env as Environment<T, *>).asSequence().sorted()
            .dropWhile { it.id != fromId }
            .take(numberOfNodes)
            .map { incarnation.getProperty(it, molecule, property) }
            .toList().toDoubleArray()
            .takeIf { it.size == numberOfNodes }
            ?: throw IllegalStateException("Not enough nodes to export data from")

    override fun getNames() = generateSequence(fromId) { it + 1 }
        .take(numberOfNodes)
        .map { "$molecule@node$fromId+${if (it == 0) "" else it}" }
        .toList()

}