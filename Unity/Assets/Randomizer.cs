using System;
using System.IO;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;
using System.Reflection;

public class Randomizer : MonoBehaviour {
    TrainingImageQuality qualityControl;

    [Header("References")]
    [Tooltip("All gates in the pool. Disabled gates are hidden; enabled are visible.")]
    public List<Transform> gatePool = new List<Transform>();

    [Tooltip("Renderers of the gates, in the same order as gatePool. Used for color randomization.")]
    public List<Renderer> gateRenderers = new List<Renderer>();

    public Image bgImage;

    [Tooltip("The origin point the cone extends from (usually the camera).")]
    public Transform coneOrigin;

    [Header("Multi-Gate")]
    [Tooltip("Probability weights for visible-gate count. Index = count.")]
    public float[] gateCountWeights = new float[] { 0.05f, 0.60f, 0.25f, 0.10f };

    [Tooltip("How far behind the first gate the next ones are placed along the cone axis.")]
    public Vector2 followGateSpacing = new Vector2(4f, 12f);

    [Tooltip("How much lateral offset additional gates can have from the first.")]
    public float followGateLateralOffset = 3f;

    [Header("Cone Parameters")]
    [Range(0f, 89f)]
    public float coneHalfAngleDegrees = 30f;
    public float minDistance = 3f;
    public float maxDistance = 20f;

    [Header("Rotation Parameters")]
    [Range(0f, 180f)] public float maxYawDeviation = 60f;
    [Range(0f, 45f)] public float maxRoll = 15f;
    [Range(0f, 45f)] public float maxPitch = 15f;

    [Header("Lighting Randomization")]
    public Light directionalLight;
    public Vector2 intensityRange = new Vector2(0.5f, 2.5f);
    public Vector2 pitchRange = new Vector2(15f, 80f);
    [Range(0f, 1f)] public float colorVariation = 0.3f;

    [Header("Occlusion")]
    [Range(0f, 1f)]
    public float occlusionProbability = 0.3f;

    [Tooltip("Max number of occluding objects per sample.")]
    public int maxOccluders = 3;

    [Tooltip("How big occluders can be (random size on each axis).")]
    public Vector2 occluderSizeRange = new Vector2(0.3f, 1.5f);

    [SerializeField] List<GameObject> occluderPool = new List<GameObject>();

    string[] cachedFiles;

    [SerializeField] float jitterScaler = 0.15f;

    [Header("Batch Generation")]
    [Tooltip("If true, keeps calling Randomize() every frame in Play mode until targetCount images have been saved.")]
    public bool batchMode = false;
    [Tooltip("Target number of saved images for this session.")]
    public int targetCount = 15000;
    [Tooltip("Log progress every N saved frames.")]
    public int logEvery = 100;




    void Awake() {
        string imagesFolder = @"C:\Users\matt\Downloads\val2017\val2017";
        cachedFiles = Directory.GetFiles(imagesFolder, "*.jpg");
    }


    void Start() {
        if (coneOrigin == null) coneOrigin = Camera.main?.transform;
        EnsureGateColliders();
        Randomize();
    }

    private float batchStartTime;
    private int batchStartCount;
    private int lastLoggedCount;

    void Update() {
        if (!batchMode) return;
        if (qualityControl == null) qualityControl = GetComponent<TrainingImageQuality>();
        if (qualityControl == null) return;

        if (batchStartTime == 0f) {
            batchStartTime = Time.realtimeSinceStartup;
            batchStartCount = qualityControl.savedFrameCount;
            lastLoggedCount = batchStartCount;
        }

        if (qualityControl.savedFrameCount >= targetCount) {
            float elapsed = Time.realtimeSinceStartup - batchStartTime;
            int produced = qualityControl.savedFrameCount - batchStartCount;
            Debug.Log($"Batch complete: {produced} frames in {elapsed:F1}s ({produced / elapsed:F1} fps). Total saved: {qualityControl.savedFrameCount}");
            batchMode = false;
            batchStartTime = 0f;
            return;
        }

        Randomize();

        if (qualityControl.savedFrameCount - lastLoggedCount >= logEvery) {
            lastLoggedCount = qualityControl.savedFrameCount;
            float elapsed = Time.realtimeSinceStartup - batchStartTime;
            int produced = qualityControl.savedFrameCount - batchStartCount;
            float fps = produced / elapsed;
            float remaining = (targetCount - qualityControl.savedFrameCount) / Mathf.Max(fps, 0.01f);
            Debug.Log($"Batch: {qualityControl.savedFrameCount}/{targetCount} saved | {fps:F1} fps | ETA {remaining:F0}s");
        }
    }

    void EnsureGateColliders() {
        foreach (var gate in gatePool) {
            if (gate == null) continue;
            if (gate.GetComponentInChildren<Collider>(true) != null) continue;

            // Add a MeshCollider to whichever child holds the MeshFilter.
            var meshFilter = gate.GetComponentInChildren<MeshFilter>(true);
            if (meshFilter == null) continue;
            var mc = meshFilter.gameObject.AddComponent<MeshCollider>();
            mc.sharedMesh = meshFilter.sharedMesh;
        }
    }


    public void Randomize() {
        if (coneOrigin == null || gatePool.Count == 0) {
            Debug.LogWarning("Randomizer: missing cone origin or empty gate pool.");
            return;
        }

        if (qualityControl == null) qualityControl = GetComponent<TrainingImageQuality>();
        qualityControl.ClearLines();

        var assembly = Assembly.GetAssembly(typeof(SceneView));
        var logEntries = assembly.GetType("UnityEditor.LogEntries");
        var clearMethod = logEntries.GetMethod("Clear");
        clearMethod.Invoke(new object(), null);


        RandomizeBackground();
        RandomizeLight();
        RandomizeGates();
        RandomizeOccluders();

        for (int i = 0; i < gatePool.Count; i++) {
            RandomizeGateColor(i);
        }

        foreach(var x in gatePool) {
            if (!x.gameObject.activeSelf) continue;
            bool result = qualityControl.ComputeVisibleFraction(x);
        }

        qualityControl.FlushTrainingFrame();
        if (!Application.isPlaying) FindObjectOfType<KeypointVisualizer>()?.ShowLatest();
    }

    void EnsureOccluderPool() {
        // Adopt or destroy any leftover occluders from previous sessions/recompiles.
        if (occluderPool.Count == 0) {
            foreach (var existing in GameObject.FindObjectsOfType<GameObject>()) {
                if (existing.name != "Occluder") continue;
                if (occluderPool.Count < maxOccluders) {
                    existing.SetActive(false);
                    occluderPool.Add(existing);
                } else {
                    DestroyImmediate(existing);
                }
            }
        }

        // Trim if maxOccluders was lowered.
        while (occluderPool.Count > maxOccluders) {
            int last = occluderPool.Count - 1;
            if (occluderPool[last] != null) DestroyImmediate(occluderPool[last]);
            occluderPool.RemoveAt(last);
        }

        while (occluderPool.Count < maxOccluders) {
            PrimitiveType[] types = { PrimitiveType.Cube, PrimitiveType.Sphere,
                                  PrimitiveType.Capsule, PrimitiveType.Cylinder };
            GameObject obj = GameObject.CreatePrimitive(types[Random.Range(0, types.Length)]);
            obj.name = "Occluder";

            // Replace the type-specific primitive collider with a convex MeshCollider so non-uniform
            // scale on Spheres/Capsules/Cylinders produces a collider matching the visible mesh.
            var oldCollider = obj.GetComponent<Collider>();
            if (oldCollider != null) DestroyImmediate(oldCollider);
            var mf = obj.GetComponent<MeshFilter>();
            if (mf != null) {
                var mc = obj.AddComponent<MeshCollider>();
                mc.sharedMesh = mf.sharedMesh;
                mc.convex = true;
            }

            obj.SetActive(false);
            occluderPool.Add(obj);
        }
    }


    public void RandomizeOccluders() {
        EnsureOccluderPool();

        // Hide all first
        foreach (var occ in occluderPool)
            occ.SetActive(false);

        var visibleGates = gatePool.FindAll(g => g.gameObject.activeSelf);

        int occluderIndex = 0;
        foreach (var gate in visibleGates) {
            if (occluderIndex >= occluderPool.Count) break;

            // Roll the dice � only occlude some of the time
            if (Random.value > occlusionProbability) continue;

            // Pick a random point on the gate's face (2.7m x 2.7m frontal)
            Vector3 gateLocalOffset = new Vector3(
                Random.Range(-1.35f, 1.35f),
                Random.Range(-1.35f, 1.35f),
                0f
            );
            Vector3 gatePoint = gate.TransformPoint(gateLocalOffset);

            // Place occluder somewhere along the line from camera to that point
            float t = Random.Range(0.3f, 0.85f);
            Vector3 basePos = coneOrigin.position + (gatePoint - coneOrigin.position) * t;

            // Small jitter
            Vector3 jitter = Random.insideUnitSphere * 0.3f;

            var occ = occluderPool[occluderIndex];
            occ.SetActive(true);
            occ.transform.position = basePos + jitter;
            occ.transform.localScale = new Vector3(
                Random.Range(occluderSizeRange.x, occluderSizeRange.y),
                Random.Range(occluderSizeRange.x, occluderSizeRange.y),
                Random.Range(occluderSizeRange.x, occluderSizeRange.y)
            );
            occ.transform.rotation = Random.rotation;

            // Random color
            var renderer = occ.GetComponent<Renderer>();
            if (renderer != null) {
                Color randomColor = new Color(Random.value, Random.value, Random.value, 1f);
                var props = new MaterialPropertyBlock();
                renderer.GetPropertyBlock(props);
                props.SetColor("_BaseColor", randomColor);
                renderer.SetPropertyBlock(props);
            }

            occluderIndex++;
        }

        // Push transform changes into the physics scene so raycasts see the new collider positions.
        Physics.SyncTransforms();
    }


    void RandomizeGates() {
        int gateCount = SampleGateCount();

        // Hide all first
        foreach (var gate in gatePool)
            gate.gameObject.SetActive(false);

        // Place each active gate independently in the cone
        for (int i = 0; i < gateCount && i < gatePool.Count; i++) {
            PlaceGateInCone(i);
            RandomizeGateColor(i);
        }
    }


    void PlaceGateInCone(int gateIndex) {
        var gate = gatePool[gateIndex];
        gate.gameObject.SetActive(true);

        // Random distance along cone axis
        float distance = Random.Range(minDistance, maxDistance);

        // Uniform spherical-cap sampling for direction
        float cosHalfAngle = Mathf.Cos(coneHalfAngleDegrees * Mathf.Deg2Rad);
        float cosTheta = Random.Range(cosHalfAngle, 1f);
        float sinTheta = Mathf.Sqrt(1f - cosTheta * cosTheta);
        float phi = Random.Range(0f, 2f * Mathf.PI);

        Vector3 localDir = new Vector3(
            sinTheta * Mathf.Cos(phi),
            sinTheta * Mathf.Sin(phi),
            cosTheta
        );

        Vector3 worldDir = coneOrigin.TransformDirection(localDir);
        gate.position = coneOrigin.position + worldDir * distance;

        // Face the origin, with random tilt
        Quaternion baseRot = Quaternion.LookRotation(coneOrigin.position - gate.position, Vector3.up);
        Quaternion offsetRot = Quaternion.Euler(
            Random.Range(-maxPitch, maxPitch),
            Random.Range(-maxYawDeviation, maxYawDeviation),
            Random.Range(-maxRoll, maxRoll)
        );
        gate.rotation = baseRot * offsetRot;
    }


    int SampleGateCount() {
        // Weighted random selection from gateCountWeights array
        float total = 0f;
        foreach (var w in gateCountWeights) total += w;

        float roll = Random.value * total;
        float cumulative = 0f;
        for (int i = 0; i < gateCountWeights.Length; i++) {
            cumulative += gateCountWeights[i];
            if (roll < cumulative) return i;
        }
        return gateCountWeights.Length - 1;
    }


    void RandomizeGateColor(int gateIndex) {
        if (gateIndex >= gateRenderers.Count || gateRenderers[gateIndex] == null) return;

        Color randomColor = Color.HSVToRGB(
            Random.value,
            Random.Range(0.6f, 1.0f),
            Random.Range(0.7f, 1.0f)
        );

        var props = new MaterialPropertyBlock();
        gateRenderers[gateIndex].GetPropertyBlock(props);
        props.SetColor("_BaseColor", randomColor); // URP
        props.SetColor("_Color", randomColor);     // Built-in RP
        gateRenderers[gateIndex].SetPropertyBlock(props);
    }


    public void RandomizeLight() {
        if (directionalLight == null) return;

        directionalLight.intensity = Random.Range(intensityRange.x, intensityRange.y);

        float yaw = Random.Range(0f, 360f);
        float pitch = Random.Range(pitchRange.x, pitchRange.y);
        directionalLight.transform.rotation = Quaternion.Euler(pitch, yaw, 0f);

        float hue = Random.Range(0.05f, 0.18f);
        if (Random.value < 0.5f) hue = Random.Range(0.55f, 0.7f);

        float saturation = Random.Range(0f, colorVariation);
        directionalLight.color = Color.HSVToRGB(hue, saturation, 1f);

        directionalLight.shadowStrength = Random.Range(0.5f, 1.0f);
    }


    public void RandomizeBackground() {
        if (cachedFiles == null || cachedFiles.Length == 0) Awake();

        Camera.main.backgroundColor = new Color(Random.value, Random.value, Random.value, 1f);

        if (Random.value < 0.25f) {
            bgImage.enabled = false;
            return;
        }

        bgImage.enabled = true;

        string randomPath = cachedFiles[Random.Range(0, cachedFiles.Length)];
        byte[] data = File.ReadAllBytes(randomPath);
        Texture2D imageTex = new Texture2D(2, 2);
        imageTex.LoadImage(data);

        Sprite imageSprite = Sprite.Create(
            imageTex,
            new Rect(0, 0, imageTex.width, imageTex.height),
            new Vector2(0.5f, 0.5f)
        );
        bgImage.sprite = imageSprite;
    }


    // --- Editor visualization ----------------------------------------------
    void OnDrawGizmosSelected() {
        if (coneOrigin == null) return;
        Gizmos.color = Color.yellow;
        Gizmos.matrix = coneOrigin.localToWorldMatrix;
        float r_near = minDistance * Mathf.Tan(coneHalfAngleDegrees * Mathf.Deg2Rad);
        float r_far = maxDistance * Mathf.Tan(coneHalfAngleDegrees * Mathf.Deg2Rad);
        for (int i = 0; i < 16; i++) {
            float angle = i * (Mathf.PI * 2 / 16);
            Vector3 near = new Vector3(Mathf.Cos(angle) * r_near, Mathf.Sin(angle) * r_near, minDistance);
            Vector3 far = new Vector3(Mathf.Cos(angle) * r_far, Mathf.Sin(angle) * r_far, maxDistance);
            Gizmos.DrawLine(near, far);
        }
        DrawCircle(minDistance, r_near);
        DrawCircle(maxDistance, r_far);
        Gizmos.matrix = Matrix4x4.identity;
    }

    void DrawCircle(float z, float radius) {
        const int segments = 32;
        Vector3 prev = new Vector3(radius, 0, z);
        for (int i = 1; i <= segments; i++) {
            float angle = i * (Mathf.PI * 2 / segments);
            Vector3 next = new Vector3(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius, z);
            Gizmos.DrawLine(prev, next);
            prev = next;
        }
    }
}


[AttributeUsage(AttributeTargets.Method)]
public class InspectorButtonAttribute : Attribute {
    public string Label;
    public InspectorButtonAttribute(string label = null) { Label = label; }
}


[CustomEditor(typeof(Randomizer))]
public class RandomizeGatePoseEditor : Editor {
    public override void OnInspectorGUI() {
        DrawDefaultInspector();
        if (GUILayout.Button("Randomize Now")) {
            ((Randomizer)target).Randomize();
        }
    }
}