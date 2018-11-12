import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import org.apache.tools.ant.taskdefs.condition.Os
import java.io.ByteArrayOutputStream

plugins {
    java
    kotlin("jvm") version "1.3.0"
}

repositories {
    mavenCentral()
}

dependencies {
    api("it.unibo.alchemist:alchemist:8.0.0-beta+0q8.31ea0")
    implementation(kotlin("stdlib-jdk8"))
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}
tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

sourceSets.getByName("main") {
    resources {
        srcDirs("src/main/protelis")
    }
}

task("runTests") {
    doLast {
        println("Done.")
    }
}
fun makeTest(
    file: String,
    sampling: Double = 1.0,
    time: Double = Double.POSITIVE_INFINITY,
    vars: Set<String> = setOf(),
    maxHeap: Long? = null,
    threads: Int? = null,
    debug: Boolean = false
) {
    val heap: Long = maxHeap ?: if (System.getProperty("os.name").toLowerCase().contains("linux")) {
        ByteArrayOutputStream().use { output ->
            exec {
                executable = "bash"
                args = listOf("-c", "cat /proc/meminfo | grep MemTotal | grep -o '[0-9]*'")
                standardOutput = output
            }
            output.toString().trim().toLong()
        } / 1024 - 2 * 1024
    } else {
        // Guess 16GB RAM of which 2 used by the OS
        14 * 1024L
    }
    val threadCount = threads ?: Math.min(Runtime.getRuntime().availableProcessors(), heap.toInt() / 2 / 1024)
    println("Running on $threads threads")
    task<JavaExec>("run$file") {
        classpath = sourceSets["main"].runtimeClasspath
        classpath("src/main/protelis")
        main = "it.unibo.alchemist.Alchemist"
        maxHeapSize = "${heap}m"
        if (debug) {
            jvmArgs("-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=1044")
        }
        File("data").mkdirs()
        args(
            "-y", "src/main/yaml/${file}.yml",
            "-t", "$time",
            "-e", "data/${file}",
            "-i", "$sampling"
        )
        if (vars.isNotEmpty()) {
            args("-b", "-var", *vars.toTypedArray())
        }
    }
    tasks {
        "runTests" {
            dependsOn("run$file")
        }
    }
}
makeTest(file = "vienna", sampling = 5.0)
defaultTasks("runTests")
