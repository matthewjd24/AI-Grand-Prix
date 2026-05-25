using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

/// <summary>
/// Builds the per-frame keypoint JSON written next to each training image.
/// Pure annotation/serialization — kept separate from the quality check.
/// </summary>
public class TrainingLabelWriter : MonoBehaviour {
    [Tooltip("Axis (in gate-local space) along which the gate's depth runs. Used to mirror corner Transforms across the gate's center plane so they sit on the camera-facing face.")]
    public Vector3 gateDepthAxisLocal = new Vector3(0f, 0f, 1f);

    [Tooltip("When true, also export the 4 OUTER frame corners (TL_out/TR_out/BL_out/BR_out child Transforms) for 8-keypoint training. When false, only the inner 4 are written.")]
    public bool includeOuterCorners = false;

    // Order is preserved in the output JSON and MUST match _GATE_PTS_3D in predict.py.
    private static readonly string[] InnerCornerNames = { "TL", "TR", "BL", "BR" };
    private static readonly string[] OuterCornerNames = { "TL_out", "TR_out", "BL_out", "BR_out" };

    private string[] ActiveCornerNames =>
        includeOuterCorners
            ? new[] { "TL", "TR", "BL", "BR", "TL_out", "TR_out", "BL_out", "BR_out" }
            : InnerCornerNames;

    private List<GateAnnotation> frameGates = new List<GateAnnotation>();

    /// <summary>Build annotations for every gate in the list. Replaces any
    /// annotations previously buffered for this frame.</summary>
    public void WriteAll(IEnumerable<Transform> gates, Camera cam) {
        frameGates.Clear();
        foreach (var gate in gates) {
            frameGates.Add(BuildAnnotation(gate, cam));
        }
    }

    /// <summary>Clear the buffered annotations (e.g. for empty frames).</summary>
    public void Clear() {
        frameGates.Clear();
    }

    GateAnnotation BuildAnnotation(Transform gate, Camera cam) {
        var names = ActiveCornerNames;
        var ann = new GateAnnotation {
            gate = gate.name,
            corners = new CornerKeypoint[names.Length],
        };

        Vector3 gateCenterWorld = GetComponent<Renderer>() != null ? GetComponent<Renderer>().bounds.center : gate.position;
        Vector3 depthNormalWorld = gate.TransformDirection(gateDepthAxisLocal).normalized;

        for (int i = 0; i < names.Length; i++) {
            Transform cornerT = FindCornerChild(gate, names[i]);
            var kp = new CornerKeypoint { name = names[i] };
            if (cornerT == null) {
                kp.present = false;
                ann.corners[i] = kp;
                continue;
            }
            kp.present = true;

            // Reflect corner across the gate's center plane along its depth normal;
            // pick whichever copy is closer to the camera (= front face).
            Vector3 cornerWorld = cornerT.position;
            float signedDist = Vector3.Dot(cornerWorld - gateCenterWorld, depthNormalWorld);
            Vector3 mirrored = cornerWorld - 2f * signedDist * depthNormalWorld;
            // Choose whichever copy is closer to the camera (= front face).
            // CRITICAL: do NOT write `chosen` back to cornerT.position. That
            // would permanently move the corner Transform every frame and
            // accumulate drift over thousands of randomizations.
            Vector3 world = (cornerWorld - cam.transform.position).sqrMagnitude
                          < (mirrored - cam.transform.position).sqrMagnitude
                          ? cornerWorld : mirrored;
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

    public string BuildFrameJson(string imageFilename, int width, int height) {
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

    [Serializable]
    class CornerKeypoint {
        public string name;
        public bool present;
        public bool onScreen;
        public bool visible;
        public float x, y;
    }

    [Serializable]
    class GateAnnotation {
        public string gate;
        public CornerKeypoint[] corners;
    }
}
