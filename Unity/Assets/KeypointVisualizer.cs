using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class KeypointVisualizer : MonoBehaviour {
    [Tooltip("Drag 12 UI Images here, grouped per gate: [Gate0 TL,TR,BL,BR, Gate1 TL,TR,BL,BR, Gate2 TL,TR,BL,BR].")]
    public RectTransform[] markers = new RectTransform[12];

    public Color visibleColor = Color.green;
    public Color occludedColor = Color.red;
    public Color offscreenColor = Color.yellow;

    [Tooltip("Folder (relative to project root) holding the JSON sidecars.")]
    public string folder = "TrainingImages";

    public void ShowLatest() {
        string path = Path.Combine(Application.dataPath, "..", folder);
        var dir = new DirectoryInfo(path);
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
        for (int g = 0; g < data.gates.Length; g++) {
            var gate = data.gates[g];
            for (int i = 0; i < 4; i++) {
                int slot = g * 4 + i;
                if (slot >= markers.Length || markers[slot] == null) continue;
                if (i >= gate.corners.Length || !gate.corners[i].present) continue;

                var c = gate.corners[i];
                markers[slot].gameObject.SetActive(true);
                // JSON: top-left origin in image pixels. Canvas overlay: bottom-left origin in screen pixels.
                float xUI = cr.x + c.x * sx;
                float yUI = cr.y + (cr.height - c.y * sy);
                markers[slot].anchoredPosition = new Vector2(xUI, yUI);

                var img = markers[slot].GetComponent<Image>();
                if (img != null) img.color = !c.onScreen ? offscreenColor : (c.visible ? visibleColor : occludedColor);
            }
        }
    }

    public void Hide() {
        foreach (var m in markers) if (m != null) m.gameObject.SetActive(false);
    }

    [System.Serializable]
    class CornerKeypoint { public string name; public bool present, onScreen, visible; public float x, y; }

    [System.Serializable]
    class GateAnnotation { public string gate; public float visibleFraction; public CornerKeypoint[] corners; }

    [System.Serializable]
    class FrameAnnotation { public string image; public int width, height; public GateAnnotation[] gates; }
}
