import groovy.json.JsonSlurper
import java.security.MessageDigest

plugins {
    id("com.android.application")
}

// The inference session factory and logger are part of this standalone application.
val depthRuntime = JsonSlurper().parse(rootProject.file("testdata/runtime-uint16.json")) as Map<*, *>
val depthOrtVersion = depthRuntime["ort"] as String
val depthQairtVersion = depthRuntime["qairt"] as String
val depthRuntimeId = MessageDigest.getInstance("SHA-256").digest(
    (depthRuntime["files"] as Map<*, *>).entries.sortedBy { it.key.toString() }
        .joinToString("|") { "${it.key}=${it.value}" }.toByteArray(Charsets.UTF_8)
).take(8).joinToString("") { "%02x".format(it.toInt() and 0xff) }


android {
    namespace = "com.yy26git.depthanythingv2qnn"
    compileSdk = 35

    // Pin the NDK used for the validated QAIRT 2.50 / HTP V68 runtime set.
    ndkVersion = "27.2.12479018"

    buildFeatures { buildConfig = true }
    sourceSets.getByName("androidTest").assets.srcDir("src/test/resources")

    defaultConfig {
        applicationId = "com.yy26git.depthanythingv2qnn"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        buildConfigField("String", "DEPTH_ORT_VERSION", "\"$depthOrtVersion\"")
        buildConfigField("String", "DEPTH_QAIRT_VERSION", "\"$depthQairtVersion\"")
        buildConfigField("String", "DEPTH_RUNTIME_ID", "\"$depthRuntimeId\"")

        // The preview stage ships instrumentation tests, so the module needs a runner declared.
        // Nothing about it changes the release APK: androidTest sources are not packaged.
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake { arguments += "-DANDROID_STL=c++_static" }
        }

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
            isJniDebuggable = true
        }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    // The QNN runtime libraries are hand-staged in src/main/jniLibs/arm64-v8a. They are NOT
    // extracted from the ONNX Runtime AAR: the staged runtime set is a matched set for QAIRT
    // 2.50 / HTP V68, and mixing it with whatever ships inside the AAR is how mismatched
    // libQnnHtp / libQnnHtpV68 pairings happen.
    //
    // useLegacyPackaging must stay TRUE. It is not a style choice:
    // OnnxSessionFactory resolves the QNN runtime by listing
    // `context.applicationInfo.nativeLibraryDir`, which is only populated when
    // extractNativeLibs=true. With useLegacyPackaging=false the device reports
    // extractNativeLibs=false, nativeLibraryDir is empty, and every QNN library reads as
    // "missing" at runtime even though the APK contains all eight. The app fails loudly on
    // exactly that condition ("Missing Qualcomm QNN HTP runtime libraries"), which is how it was
    // caught.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    // Official pinned QNN-enabled ORT. The session factory owns backend configuration;
    // artifact hashes and the typed bridge ABI are checked before each build.
    implementation(files("libs/onnxruntime-android-qnn-$depthOrtVersion.aar"))

    // CameraX 1.5.3 — the newest line that resolves against this module, and that was verified
    // rather than assumed. Each 1.5.3 AAR declares, in
    // META-INF/com/android/build/gradle/aar-metadata.properties:
    //
    //   minCompileSdk=35                matches compileSdk = 35 exactly
    //   minAndroidGradlePluginVersion=8.6.0   satisfied by AGP 9.2.1
    //   <uses-sdk android:minSdkVersion="23">  satisfied by minSdk = 26
    //
    // CameraX 1.6.x was checked and rejected: its AAR metadata declares minCompileSdk=36, so
    // 1.6.0/1.6.1/1.6.2 fail dependency verification here unless compileSdk is raised to 36 —
    // which is a platform change to this harness, not a camera decision. 1.5.3 is therefore the
    // ceiling, not a preference. camera-view 1.5.3 is what supplies PreviewView and its
    // setFrameUpdateListener, which is how the instrumentation tests observe arriving frames.
    //
    // All four artifacts are pinned to the same version. CameraX requires camera-core,
    // camera-camera2, camera-lifecycle and camera-view to agree; a mixed set resolves to the
    // highest version transitively and can leave duplicate classes at runtime.
    implementation("androidx.camera:camera-core:1.5.3")
    implementation("androidx.camera:camera-camera2:1.5.3")
    implementation("androidx.camera:camera-lifecycle:1.5.3")
    implementation("androidx.camera:camera-view:1.5.3")

    // MainActivity moves from android.app.Activity to ComponentActivity because CameraX binds to a
    // androidx LifecycleOwner, which the platform Activity is not. 1.9.3 is already in this
    // machine's Gradle cache (the policy the root build file states) and declares
    // minCompileSdk=34, so it resolves against compileSdk 35 without a second cold download.
    implementation("androidx.activity:activity-ktx:1.9.3")

    // Host-side JVM unit tests. Never packaged into the harness APK, exactly like the
    // androidTest set below. Without this the module declares testInstrumentationRunner but has
    // no unit-test classpath, so `testDebugUnitTest` fails to compile the test source set rather
    // than running it — a green build that has silently never executed a test. Pinned to the same
    // 4.13.2 the instrumentation tests use so there is one JUnit version in the project.
    testImplementation("junit:junit:4.13.2")

    // Instrumentation only — never packaged into the harness APK. Versions are the current stable
    // androidx.test line: runner 1.6.2 and core 1.6.1 both carry <uses-sdk android:minSdkVersion="19">
    // and declare no minCompileSdk in their AAR metadata, so unlike CameraX 1.6 they cannot raise
    // this module's compileSdk floor. Declared are exactly the artifacts the tests touch.
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:core:1.6.1")   // ActivityScenario
    androidTestImplementation("androidx.test.ext:junit:1.2.1")  // AndroidJUnit4
    androidTestImplementation("junit:junit:4.13.2")
}

val verifyDepthRuntime by tasks.registering(Exec::class) {
    workingDir(rootProject.projectDir)
    commandLine("python3", "scripts/verify-runtime-artifacts.py")
}
tasks.named("preBuild").configure { dependsOn(verifyDepthRuntime) }
