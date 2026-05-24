using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class KeypointVisualizer : MonoBehaviour {
    [Tooltip("UI Images grouped per gate. With 4-keypoint labels: gate0 TL,TR,BL,BR, gate1 …. " +
             "With 8-keypoint labels: gate0 TL,TR,BL,BR,TL_out,TR_out,BL_out,BR_out, gate1 …. " +
             "Stride is read from the JSON's corners-per-gate count, so size this array as " +
             "maxGates * maxCornersPerGate (e.g. 24 = 3 gates * 8 corners).")]
    public RectTransform[] markers = new RectTransform[24];

    public Color visibleColor = Color.green;
    public Color occludedColor = Color.red;
    public Color offscreenColor = Color.yellow;
    public Color outerVisibleColor = Color.cyan;
    public Color outerOccludedColor = new Color(0.6f, 0f, 0.6f);

    [Tooltip("Absolute path to the folder holding the JSON sidecars. Must match Randomizer's screenshotFolder.")]
    public string folder = @"C:\Users\matt\Documents\GitHub\AI-Grand-Prix\TrainingImages";

    public void ShowLatestKeypoints() {
        var dir = new DirectoryInfo(folder);
        if (!dir.Exists) return;
        var latest = dir.GetFiles("*.json").OrderByDescending(f => f.LastWriteTime).FirstOrDefault();
        if (latest == null) { Hide(); return; }
        ShowFromJson(latest.FullName);
    }

    public void ShowFromJson(string jsonPath) {
        if (!File.Exists(jsonPath)) return;

        var data = JsonUtility.FromJson<FrameAnnotation>(File.ReadAllText(jsonPath));
        if (data == null || data.gates == null || data.gates.Length == 0) { Hide(); return; }

        // Map JSON coords into the camera's actual on-screen rect, not the whole canvas.
        // This handles letterboxing when the Game View aspect != camera aspect.
        var cam = Camera.main;
        if (cam == null) return;
        Rect cr = cam.pixelRect; // bottom-left origin, in screen pixels
        float sx = data.width  > 0 ? cr.width  / data.width  : 1f;
        float sy = data.height > 0 ? cr.height / data.height : 1f;

        Hide();

        // Stride = corners per gate. Read from the first gate so the same
        // visualizer handles 4-keypoint and 8-keypoint datasets without
        // re-configuration.
        int stride = data.gates[0].corners != null ? data.gates[0].corners.Length : 0;

        // Debug summary: for each gate, count how many corners are present.
        var counts = new System.Text.StringBuilder();
        for (int gi = 0; gi < data.gates.Length; gi++) {
            int present = 0;
            if (data.gates[gi].corners != null) {
                foreach (var c in data.gates[gi].corners) if (c.present) present++;
            }
            if (gi > 0) counts.Append(", ");
            counts.AppendFormat("g{0}={1}/{2}", gi, present, data.gates[gi].corners?.Length ?? 0);
        }
        // Debug.Log($"[KeypointVisualizer] {System.IO.Path.GetFileName(jsonPath)} stride={stride} gates={data.gates.Length} corners[{counts}]");

        if (stride == 0) return;

        int shown = 0, skippedSlotMissing = 0, skippedNotPresent = 0;
        for (int g = 0; g < data.gates.Length; g++) {
            var gate = data.gates[g];
            for (int i = 0; i < gate.corners.Length; i++) {
                int slot = g * stride + i;
                if (slot >= markers.Length || markers[slot] == null) {
                    skippedSlotMissing++;
                    continue;
                }
                if (!gate.corners[i].present) {
                    skippedNotPresent++;
                    continue;
                }

                var c = gate.corners[i];
                markers[slot].gameObject.SetActive(true);
                // JSON: top-left origin in image pixels. Canvas overlay: bottom-left origin in screen pixels.
                float xUI = cr.x + c.x * sx;
                float yUI = cr.y + (cr.height - c.y * sy);
                markers[slot].anchoredPosition = new Vector2(xUI, yUI);

                var img = markers[slot].GetComponent<Image>();
                if (img != null) img.color = ColorFor(c, IsOuter(c.name));
                shown++;
            }
        }
        // Debug.Log($"[KeypointVisualizer] shown={shown}, slotMissing={skippedSlotMissing}, notPresent={skippedNotPresent}, markersArray={markers.Length}");
    }

    static bool IsOuter(string name) => name != null && name.EndsWith("_out");

    Color ColorFor(CornerKeypoint c, bool outer) {
        if (!c.onScreen) return offscreenColor;
        if (outer)       return c.visible ? outerVisibleColor : outerOccludedColor;
        return c.visible ? visibleColor : occludedColor;
    }

    public void Hide() {
        foreach (var m in markers) if (m != null) m.gameObject.SetActive(false);
    }

    [System.Serializable]
    class CornerKeypoint { public string name; public bool present, onScreen, visible; public float x, y; }

    [System.Serializable]
    class GateAnnotation { public string gate; public CornerKeypoint[] corners; }

    [System.Serializable]
    class FrameAnnotation { public string image; public int width, height; public GateAnnotation[] gates; }
}
