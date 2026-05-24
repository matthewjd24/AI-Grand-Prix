using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Quality gate for training images. Single entrypoint: CheckAllGates.
///
/// Returns the gates that passed visibility, or null if any active gate failed
/// (= frame is rejected). An empty (but non-null) result means no gates were
/// active — caller may treat that as a negative-example frame.
///
/// MonoBehaviour only because OnDrawGizmos needs a component to live on; the
/// class carries no externally-visible state between calls.
/// </summary>
public class TrainingImageQuality : MonoBehaviour {
    private List<(Vector3 start, Vector3 end, Color color)> debugLines
        = new List<(Vector3, Vector3, Color)>();
    [SerializeField] float gateVisibilityThreshold;

    /// <summary>Run the visibility check on every active gate in <paramref name="gates"/>.
    /// Returns the list of gates that passed, or null if ANY active gate failed.</summary>
    public List<Transform> CheckAllGates(IEnumerable<Transform> gates) {
        debugLines.Clear();
        var passed = new List<Transform>();
        foreach (var gate in gates) {
            if (gate == null || !gate.gameObject.activeSelf) continue;
            if (IsGateVisible(gate)) passed.Add(gate);
            else                    return null;     // any failure rejects the whole frame
        }
        return passed;
    }

    bool IsGateVisible(Transform gate) {
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

        if (total == 0) return false;
        return (float)visible / total > gateVisibilityThreshold;
    }

    void OnDrawGizmos() {
        foreach (var line in debugLines) {
            Gizmos.color = line.color;
            Gizmos.DrawLine(line.start, line.end);
        }
    }
}
