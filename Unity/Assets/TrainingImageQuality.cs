using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class TrainingImageQuality : MonoBehaviour {
    [Tooltip("Folder (relative to project root) where pass screenshots are saved.")]
    public string screenshotFolder = "TrainingImages";

    [Tooltip("Folder used when validationSet is true.")]
    public string validationFolder = "ValidationImages";

    [Tooltip("Route saved frames to validationFolder instead of screenshotFolder.")]
    public bool validationSet = false;

    [Tooltip("Axis (in gate-local space) along which the gate's depth runs. Usually Z (0,0,1). Used to mirror corner Transforms across the gate's center plane so they sit on the camera-facing face.")]
    public Vector3 gateDepthAxisLocal = new Vector3(0f, 0f, 1f);

    [Tooltip("Also save frames where no gates were active (negative training examples).")]
    public bool saveEmptyFrames = true;

    [Tooltip("Number of frames actually written to disk this session. Read-only.")]
    public int savedFrameCount = 0;

    private List<(Vector3 start, Vector3 end, Color color)> debugLines
        = new List<(Vector3, Vector3, Color)>();

    // Per-frame buffer of gate annotations, flushed by FlushTrainingFrame().
    private List<GateAnnotation> frameGates = new List<GateAnnotation>();
    private bool frameHasPass = false;
    private int frameFailCount = 0;

    private static readonly string[] CornerNames = { "TL", "TR", "BL", "BR" };

    public void BeginFrame() {
        debugLines.Clear();
        frameGates.Clear();
        frameHasPass = false;
        frameFailCount = 0;
    }

    // Kept for backwards compatibility with the existing Randomizer.ClearLines() call.
    public void ClearLines() {
        BeginFrame();
    }

    public bool ComputeVisibleFraction(Transform gate) {
        const int gridSize = 8;   // 8x8 = 64 sample points
        var cam = Camera.main;

        Color visualizeColor = Color.white;
        var renderer = gate.GetComponent<Renderer>();
        if (renderer == null) renderer = gate.GetComponentInChildren<Renderer>();
        if (renderer != null) {
            var props = new MaterialPropertyBlock();
            renderer.GetPropertyBlock(props);
            visualizeColor = props.GetColor("_BaseColor");
            if (visualizeColor.a == 0f) visualizeColor = Color.white;
        }

        Vector3 sampleCenterLocal = Vector3.zero;
        if (renderer != null) {
            sampleCenterLocal = gate.InverseTransformPoint(renderer.bounds.center);
        }

        int total = 0;
        int visible = 0;

        for (int x = 0; x < gridSize; x++) {
            for (int y = 0; y < gridSize; y++) {
                float u = (x + 0.5f) / gridSize;
                float v = (y + 0.5f) / gridSize;
                Vector3 local = new Vector3(
                    sampleCenterLocal.x + Mathf.Lerp(-1.35f, 1.35f, u),
                    sampleCenterLocal.y + Mathf.Lerp(-1.35f, 1.35f, v),
                    sampleCenterLocal.z
                );
                Vector3 world = gate.TransformPoint(local);

                Vector3 screen = cam.WorldToScreenPoint(world);
                if (screen.z <= 0) continue;
                if (screen.x < 0 || screen.x >= cam.pixelWidth) continue;
                if (screen.y < 0 || screen.y >= cam.pixelHeight) continue;

                total++;

                Vector3 toPoint = world - cam.transform.position;
                float distance = toPoint.magnitude;
                bool isVisible;
                Vector3 endpoint;

                if (Physics.Raycast(cam.transform.position, toPoint.normalized,
                                    out RaycastHit hit, distance + 0.1f)) {
                    isVisible = (hit.collider.transform == gate || hit.collider.transform.IsChildOf(gate));
                    endpoint = hit.point;
                } else {
                    isVisible = true;
                    endpoint = world;
                }

                if (isVisible) visible++;

                Color drawColor = isVisible ? visualizeColor : visualizeColor * 0.3f;
                debugLines.Add((cam.transform.position, endpoint, drawColor));
            }
        }

        if (total == 0) {
            frameFailCount++;
            return false;
        }
        float fraction = (float)visible / total;
        bool passFail = fraction > 0.35f;
        // Debug.Log($"Visible fraction is : {fraction:F2}, {(passFail ? "Pass" : "Fail")}");

        if (passFail) {
            frameHasPass = true;
            frameGates.Add(BuildAnnotation(gate, fraction, cam));
        } else {
            frameFailCount++;
        }

        return passFail;
    }

    public void FlushTrainingFrame() {
        bool allPass = frameHasPass && frameFailCount == 0;
        bool empty = !frameHasPass && frameFailCount == 0; // no active gates this frame
        if (!allPass && !(empty && saveEmptyFrames)) return;

        var cam = Camera.main;
        if (cam == null) return;

        string activeFolder = validationSet ? validationFolder : screenshotFolder;
        string folder = Path.Combine(Application.dataPath, "..", activeFolder);
        Directory.CreateDirectory(folder);
        string baseName = $"pass_{System.DateTime.Now:yyyyMMdd_HHmmss_fff}";
        string pngPath = Path.Combine(folder, baseName + ".png");
        string jsonPath = Path.Combine(folder, baseName + ".json");

        CaptureCameraToPng(cam, pngPath);
        File.WriteAllText(jsonPath, BuildFrameJson(baseName + ".png", cam.pixelWidth, cam.pixelHeight));
        savedFrameCount++;
    }

    void CaptureCameraToPng(Camera cam, string path) {
        int w = cam.pixelWidth;
        int h = cam.pixelHeight;
        var rt = RenderTexture.GetTemporary(w, h, 24);
        var prevTarget = cam.targetTexture;
        var prevActive = RenderTexture.active;

        cam.targetTexture = rt;
        cam.Render();

        RenderTexture.active = rt;
        var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        tex.Apply();

        File.WriteAllBytes(path, tex.EncodeToPNG());

        cam.targetTexture = prevTarget;
        RenderTexture.active = prevActive;
        RenderTexture.ReleaseTemporary(rt);
        DestroyImmediate(tex);
    }

    GateAnnotation BuildAnnotation(Transform gate, float fraction, Camera cam) {
        var ann = new GateAnnotation {
            gate = gate.name,
            visibleFraction = fraction,
            corners = new CornerKeypoint[CornerNames.Length],
        };

        // Use the gate's renderer centroid as the mid-plane for mirroring corners.
        Vector3 gateCenterWorld = GetComponent<Renderer>() != null ? GetComponent<Renderer>().bounds.center : gate.position;
        Vector3 depthNormalWorld = gate.TransformDirection(gateDepthAxisLocal).normalized;

        for (int i = 0; i < CornerNames.Length; i++) {
            Transform cornerT = FindCornerChild(gate, CornerNames[i]);
            var kp = new CornerKeypoint { name = CornerNames[i] };
            if (cornerT == null) {
                kp.present = false;
                ann.corners[i] = kp;
                continue;
            }
            kp.present = true;

            // Reflect corner across the gate's center plane along its depth normal; pick whichever is closer to the camera.
            Vector3 cornerWorld = cornerT.position;
            float signedDist = Vector3.Dot(cornerWorld - gateCenterWorld, depthNormalWorld);
            Vector3 mirrored = cornerWorld - 2f * signedDist * depthNormalWorld;
            Vector3 chosen = (cornerWorld - cam.transform.position).sqrMagnitude
                           < (mirrored - cam.transform.position).sqrMagnitude ? cornerWorld : mirrored;
            cornerT.position = chosen;

            Vector3 world = chosen;
            Vector3 screen = cam.WorldToScreenPoint(world);
            kp.onScreen = screen.z > 0
                       && screen.x >= 0 && screen.x < cam.pixelWidth
                       && screen.y >= 0 && screen.y < cam.pixelHeight;

            kp.x = screen.x;
            kp.y = cam.pixelHeight - screen.y; // top-left origin (image convention)

            if (!kp.onScreen) {
                kp.visible = false;
            } else {
                Vector3 toPoint = world - cam.transform.position;
                float distance = toPoint.magnitude;
                if (Physics.Raycast(cam.transform.position, toPoint.normalized,
                                    out RaycastHit hit, distance + 0.1f)) {
                    kp.visible = (hit.collider.transform == gate || hit.collider.transform.IsChildOf(gate));
                } else {
                    kp.visible = true; // clear line of sight
                }
            }

            ann.corners[i] = kp;
        }
        return ann;
    }

    Transform FindCornerChild(Transform gate, string targetName) {
        foreach (var t in gate.GetComponentsInChildren<Transform>(true)) {
            if (t == gate) continue;
            if (string.Equals(t.name, targetName, StringComparison.OrdinalIgnoreCase)) return t;
        }
        return null;
    }

    string BuildFrameJson(string imageFilename, int width, int height) {
        // Hand-rolled JSON so we get nested arrays/objects (JsonUtility can't do top-level arrays cleanly).
        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.AppendFormat("  \"image\": \"{0}\",\n", imageFilename);
        sb.AppendFormat("  \"width\": {0},\n", width);
        sb.AppendFormat("  \"height\": {0},\n", height);
        sb.Append("  \"gates\": [\n");
        for (int g = 0; g < frameGates.Count; g++) {
            var ann = frameGates[g];
            sb.Append("    {\n");
            sb.AppendFormat("      \"gate\": \"{0}\",\n", EscapeJson(ann.gate));
            sb.AppendFormat(System.Globalization.CultureInfo.InvariantCulture,
                            "      \"visibleFraction\": {0:F4},\n", ann.visibleFraction);
            sb.Append("      \"corners\": [\n");
            for (int c = 0; c < ann.corners.Length; c++) {
                var kp = ann.corners[c];
                sb.Append("        {");
                sb.AppendFormat(System.Globalization.CultureInfo.InvariantCulture,
                    "\"name\":\"{0}\",\"present\":{1},\"onScreen\":{2},\"visible\":{3},\"x\":{4:F2},\"y\":{5:F2}",
                    kp.name,
                    kp.present ? "true" : "false",
                    kp.onScreen ? "true" : "false",
                    kp.visible ? "true" : "false",
                    kp.x, kp.y);
                sb.Append("}");
                if (c < ann.corners.Length - 1) sb.Append(",");
                sb.Append("\n");
            }
            sb.Append("      ]\n");
            sb.Append("    }");
            if (g < frameGates.Count - 1) sb.Append(",");
            sb.Append("\n");
        }
        sb.Append("  ]\n");
        sb.Append("}\n");
        return sb.ToString();
    }

    static string EscapeJson(string s) {
        if (s == null) return "";
        return s.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    void OnDrawGizmos() {
        foreach (var line in debugLines) {
            Gizmos.color = line.color;
            Gizmos.DrawLine(line.start, line.end);
        }
    }

    [Serializable]
    class CornerKeypoint {
        public string name;
        public bool present;   // child transform exists
        public bool onScreen;  // inside camera frustum
        public bool visible;   // not occluded
        public float x, y;     // pixel coords, top-left origin
    }

    [Serializable]
    class GateAnnotation {
        public string gate;
        public float visibleFraction;
        public CornerKeypoint[] corners;
    }
}
