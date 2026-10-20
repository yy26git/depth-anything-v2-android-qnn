package com.yy26git.depthanythingv2qnn

import android.app.Activity
import android.os.Bundle
import android.widget.TextView

/** Debug-only foreground host: the OnePlus OEM freezes background instrumentation threads.
 * Owns no camera/session, so a model gate cannot race the live pipeline's context compilation.
 */
class QnnModelTestActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(TextView(this).apply { text = "Testing depth model…" })
    }
}
