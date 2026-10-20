// Versions are pinned to the ones already present in this machine's Gradle/Maven cache,
// matching the local Android toolchain cache, so `./gradlew` resolves without a cold download.
//
// There is deliberately no `org.jetbrains.kotlin.android` plugin. AGP 9 ships built-in
// Kotlin support and registers the `kotlin` extension itself; applying the plugin on top of
// that fails with "Cannot add extension with name 'kotlin'".
plugins {
    id("com.android.application") version "9.2.1" apply false
}
