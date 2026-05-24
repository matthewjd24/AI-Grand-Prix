using System;
using System.IO;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;
using System.Reflection;

public class Randomizer : MonoBehaviour {
    TrainingImageQuality qualityControl;
    TrainingLabelWriter labelWriter;

    [Header("References")]
    [Tooltip("All gates in the pool. Disabled gates are hidden; enabled are visible.")]
    public List<Transform> gatePool = new List<Transform>();

    [Tooltip("Renderers of the gates, in the same order as gatePool. Used for color randomization.")]
    public List<Renderer> gateRenderers = new List<Renderer>();

    public Image bgImage;

    [Tooltip("The origin point the cone extends from (usually the camera).")]
    public Transform coneOrigin;

    [Header("Multi-Gate")]
    [Tooltip("Probability weights for visible-gate count. Index = count. " +
             "Skewed toward 2+ gates so inter-gate occlusion happens often.")]
    public float[] gateCountWeights = new float[] { 0.05f, 0.25f, 0.45f, 0.25f };

    [Tooltip("In multi-gate frames, the fraction where additional gates are stacked behind the first " +
             "to create real frame-on-frame occlusion. Rest are placed independently.")]
    [Range(0f, 1f)]
    public float occlusionChainProbability = 0.85f;

    [Tooltip("How far behind the lead gate the chained gates are placed (metres). Smaller values keep " +
             "the back gate's apparent size large enough to actually be occluded by the front frame, " +
             "instead of slotting through the inner hole.")]
    public Vector2 chainExtraDistance = new Vector2(0.8f, 3.5f);

    [Tooltip("Lateral viewport offset applied to chained gates. The back gate's center is pushed sideways " +
             "so its edge overlaps the front gate's frame ring rather than aligning with its hole. " +
             "Sampled as Random.Range(min, max) with a random sign.")]
    public Vector2 chainLateralOffset = new Vector2(0.05f, 0.18f);

    [Header("Placement Parameters")]
    public float minDistance = 3f;
    public float maxDistance = 20f;

    [Tooltip("Outer half-extent of the gate in metres (frame is 2.7m wide, so 1.35). " +
             "Used to compute a distance-aware viewport inset so gates can't spawn off-frame.")]
    public float gateHalfExtent = 1.35f;

    [Tooltip("Allowed off-frame fraction of the gate's apparent size. 0 = the entire gate " +
             "is guaranteed inside the viewport. 0.25 = up to a quarter of the gate may clip.")]
    [Range(0f, 1f)]
    public float allowedClipFraction = 0f;

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

    [Tooltip("Probability that occluders are skipped entirely on a normal frame.")]
    [Range(0f, 1f)]
    public float occluderSkipChance = 0.5f;

    [Tooltip("Probability that occluders are skipped on a frame where gates already occlude each other. " +
             "Higher than occluderSkipChance so we don't bury the already-occluded back gate.")]
    [Range(0f, 1f)]
    public float occluderSkipChanceWhenChained = 0.85f;

    [Tooltip("Max number of occluding objects per sample.")]
    public int maxOccluders = 3;

    [Tooltip("How big occluders can be (random size on each axis).")]
    public Vector2 occluderSizeRange = new Vector2(0.3f, 1.5f);

    [SerializeField] List<GameObject> occluderPool = new List<GameObject>();

    string[] cachedFiles;

    [SerializeField] float jitterScaler = 0.15f;

    [Header("Batch Generation")]
    [Tooltip("If true, keeps calling Randomize() every frame in Play mode until targetCount images have been saved.")]
    public bool generating = false;
    [Tooltip("Target number of saved images for this session.")]
    public int targetCount = 15000;
    [Tooltip("Log progress every N saved frames.")]
    public int logEvery = 100;

    [Tooltip("Every N saved frames, pause the batch to let the disk/GPU catch up.")]
    public int framesPerPause = 2000;
    [Tooltip("How long the batch pauses each time it hits a framesPerPause boundary, in seconds.")]
    public float pauseSeconds = 10f;

    [Header("Saving")]
    [Tooltip("Absolute path where pass screenshots are saved. Must be writable. " +
             "Absolute so builds save next to the .exe-independent location.")]
    public string screenshotFolder = @"C:\Users\matt\Documents\GitHub\AI-Grand-Prix\TrainingImages";
    [Tooltip("Absolute path used when validationSet is true.")]
    public string validationFolder = @"C:\Users\matt\Documents\GitHub\AI-Grand-Prix\ValidationImages";
    bool validationSet = false;
    [Tooltip("Also save frames where no gates were active (negative training examples).")]
    public bool saveEmptyFrames = true;
    [Tooltip("Number of frames actually written to disk this session. Read-only.")]
    public int savedFrameCount = 0;
    [Tooltip("Optional TMP text that displays the saved frame count.")]
    public TMP_Text savedCountLabel;




    void Awake() {
        string imagesFolder = @"C:\Users\matt\Downloads\val2017\val2017";
        cachedFiles = Directory.GetFiles(imagesFolder, "*.jpg");
    }


    void Start() {
        if (coneOrigin == null) coneOrigin = Camera.main?.transform;
        EnsureGateColliders();
        generating = false;   // require explicit start via UI button
        savedFrameCount = 0;
        //Randomize();
    }

    /// <summary>Wire a UI Button's OnClick to this to begin batch generation.</summary>
    public void StartBatch() {
        batchStartTime  = 0f;   // forces re-init on next Update tick
        generating       = true;
        Debug.Log($"[Randomizer] Batch started — target {targetCount} frames.");
    }

    /// <summary>Stop a running batch early.</summary>
    public void StopBatch() {
        generating = false;
        Debug.Log("[Randomizer] Batch stopped.");
    }

    /// <summary>Wire a UI Toggle's OnValueChanged(bool) to this to route saves to the validation folder.</summary>
    public void SetValidationSet(bool on) {
        validationSet = on;
        Debug.Log($"[Randomizer] Saving to {(on ? validationFolder : screenshotFolder)}");
    }

    private float batchStartTime;
    private int batchStartCount;
    private int lastLoggedCount;
    private float pauseUntil;

    void Update() {
        if (!generating) return;
        if (Time.realtimeSinceStartup < pauseUntil) return;   // currently paused
        if (qualityControl == null) qualityControl = GetComponent<TrainingImageQuality>();
        if (qualityControl == null) return;

        if (batchStartTime == 0f) {
            batchStartTime  = Time.realtimeSinceStartup;
            batchStartCount = savedFrameCount;
            lastLoggedCount = batchStartCount;
        }

        if (savedFrameCount >= targetCount) {
            float elapsed = Time.realtimeSinceStartup - batchStartTime;
            int produced  = savedFrameCount - batchStartCount;
            Debug.Log($"Batch complete: {produced} frames in {elapsed:F1}s ({produced / elapsed:F1} fps). Total saved: {savedFrameCount}");
            generating = false;
            batchStartTime = 0f;
            return;
        }

        int beforeCount = savedFrameCount;
        Randomize();

        // Periodic pause: triggered when savedFrameCount crosses a multiple of framesPerPause.
        if (framesPerPause > 0 && pauseSeconds > 0f
            && savedFrameCount / framesPerPause > beforeCount / framesPerPause) {
            pauseUntil = Time.realtimeSinceStartup + pauseSeconds;
            Debug.Log($"[Randomizer] Pausing for {pauseSeconds:F0}s at {savedFrameCount} frames.");
        }

        if (savedFrameCount - lastLoggedCount >= logEvery) {
            lastLoggedCount = savedFrameCount;
            float elapsed = Time.realtimeSinceStartup - batchStartTime;
            int produced  = savedFrameCount - batchStartCount;
            float fps = produced / elapsed;
            float remaining = (targetCount - savedFrameCount) / Mathf.Max(fps, 0.01f);
            Debug.Log($"Batch: {savedFrameCount}/{targetCount} saved | {fps:F1} fps | ETA {remaining:F0}s");
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

        // Wipe last frame's keypoint overlays so a rejected frame doesn't leave
        // stale markers from the previous accepted frame.
        FindObjectOfType<KeypointVisualizer>()?.Hide();

        if (qualityControl == null) qualityControl = GetComponent<TrainingImageQuality>();
        if (labelWriter   == null) labelWriter   = GetComponent<TrainingLabelWriter>();
        labelWriter.Clear();

#if UNITY_EDITOR
        var assembly = Assembly.GetAssembly(typeof(SceneView));
        var logEntries = assembly.GetType("UnityEditor.LogEntries");
        var clearMethod = logEntries.GetMethod("Clear");
        clearMethod.Invoke(new object(), null);
#endif


        RandomizeBackground();
        RandomizeLight();
        RandomizeGates();
        RandomizeOccluders();
        RandomizeGateColors();

        // Quality returns the list of gates that passed, or null if any failed.
        // Empty list = no gates active (negative-example frame).
        List<Transform> passed = qualityControl.CheckAllGates(gatePool);
        if (passed != null && passed.Count > 0) {
            labelWriter.WriteAll(passed, Camera.main);
            SaveTrainingFrame(passed);
            //Debug.Log("Frame passed");
            //if (!Application.isPlaying) GetComponent<KeypointVisualizer>().ShowLatestKeypoints();
        }
        else {
            //Debug.Log("Frame didn't pass QC");
        }

    }

    void SaveTrainingFrame(List<Transform> passedGates) {
        // null = a gate failed quality, skip the frame entirely.
        // empty = no gates were active (negative example) — save anyway; the
        //         JSON's empty "gates": [] is itself a valid label.
        // non-empty = all active gates passed, save with labels.
        if (passedGates == null) return;

        var cam = Camera.main;
        if (cam == null) return;

        string folder = validationSet ? validationFolder : screenshotFolder;
        Directory.CreateDirectory(folder);
        string baseName = $"pass_{System.DateTime.Now:yyyyMMdd_HHmmss_fff}";
        string pngPath  = Path.Combine(folder, baseName + ".png");
        string jsonPath = Path.Combine(folder, baseName + ".json");

        CaptureCameraToPng(cam, pngPath);
        File.WriteAllText(jsonPath, labelWriter.BuildFrameJson(baseName + ".png", cam.pixelWidth, cam.pixelHeight));
        savedFrameCount++;
        if (savedCountLabel != null) savedCountLabel.text = $"Saved: {savedFrameCount}";
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

        // Skip occluders entirely some fraction of the time. Skip much more
        // often when the gates already occlude each other, so the back gate
        // isn't buried under random clutter on top of the front gate.
        float skipChance = _frameHasGateChain ? occluderSkipChanceWhenChained : occluderSkipChance;
        if (Random.value < skipChance) {
            Physics.SyncTransforms();
            return;
        }

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

    // Set by RandomizeGates and read by RandomizeOccluders so we can dial back
    // random occluders when the gates are already occluding each other.
    private bool _frameHasGateChain;

    void RandomizeGates() {
        int gateCount = SampleGateCount();
        _frameHasGateChain = false;

        // Hide all first
        foreach (var gate in gatePool)
            gate.gameObject.SetActive(false);

        if (gateCount == 0) return;

        // Always place the first gate freely.
        PlaceGateInCone(0);

        // For multi-gate frames, decide whether to chain the rest behind the
        // first (= deliberate occlusion training) or place them independently.
        bool chain = gateCount > 1 && Random.value < occlusionChainProbability;
        _frameHasGateChain = chain;

        for (int i = 1; i < gateCount && i < gatePool.Count; i++) {
            if (chain) PlaceGateBehindLead(i, leadIndex: 0);
            else       PlaceGateInCone(i);
        }
    }

    /// <summary>Place a gate at the same viewport position as the lead, but
    /// farther from the camera, so the lead partially occludes it.</summary>
    void PlaceGateBehindLead(int gateIndex, int leadIndex) {
        var gate = gatePool[gateIndex];
        var lead = gatePool[leadIndex];
        gate.gameObject.SetActive(true);

        var cam = Camera.main;
        if (cam == null) return;

        Vector3 leadViewport = cam.WorldToViewportPoint(lead.position);

        // Pick a random direction and a magnitude in [min, max]. Random sign on each
        // axis pushes the back gate to one corner of the lead rather than
        // centering it, so its edge ends up behind the lead's solid frame ring.
        float magU = Random.Range(chainLateralOffset.x, chainLateralOffset.y) * (Random.value < 0.5f ? -1f : 1f);
        float magV = Random.Range(chainLateralOffset.x, chainLateralOffset.y) * (Random.value < 0.5f ? -1f : 1f);
        float u = Mathf.Clamp01(leadViewport.x + magU);
        float v = Mathf.Clamp01(leadViewport.y + magV);
        float distance = Mathf.Clamp(
            leadViewport.z + Random.Range(chainExtraDistance.x, chainExtraDistance.y),
            minDistance, maxDistance);

        gate.position = cam.ViewportToWorldPoint(new Vector3(u, v, distance));

        Quaternion baseRot = Quaternion.LookRotation(coneOrigin.position - gate.position, Vector3.up);
        Quaternion offsetRot = Quaternion.Euler(
            Random.Range(-maxPitch, maxPitch),
            Random.Range(-maxYawDeviation, maxYawDeviation),
            Random.Range(-maxRoll, maxRoll)
        );
        gate.rotation = baseRot * offsetRot;
    }

    void RandomizeGateColors() {
        // Count visible gates
        int visibleCount = 0;
        for (int i = 0; i < gatePool.Count; i++) {
            if (gatePool[i] != null && gatePool[i].gameObject.activeSelf) visibleCount++;
        }

        // 50/50: when there are multiple visible gates, sometimes paint them all
        // the same color so the detector can't just rely on color uniqueness.
        bool uniformColor = visibleCount > 1 && Random.value < 0.5f;

        Color sharedColor = Color.HSVToRGB(
            Random.value,
            Random.Range(0.6f, 1.0f),
            Random.Range(0.7f, 1.0f)
        );

        for (int i = 0; i < gatePool.Count; i++) {
            if (uniformColor && gatePool[i] != null && gatePool[i].gameObject.activeSelf) {
                ApplyGateColor(i, sharedColor);
            } else {
                RandomizeGateColor(i);
            }
        }
    }

    void ApplyGateColor(int gateIndex, Color color) {
        if (gateIndex >= gateRenderers.Count || gateRenderers[gateIndex] == null) return;
        var props = new MaterialPropertyBlock();
        gateRenderers[gateIndex].GetPropertyBlock(props);
        props.SetColor("_BaseColor", color); // URP
        props.SetColor("_Color", color);     // Built-in RP
        gateRenderers[gateIndex].SetPropertyBlock(props);
    }

    void PlaceGateInCone(int gateIndex) {
        var gate = gatePool[gateIndex];
        gate.gameObject.SetActive(true);

        var cam = Camera.main;
        if (cam == null) return;

        float distance = Random.Range(minDistance, maxDistance);

        // The gate's apparent half-size in viewport-Y units at this distance.
        // tan(VFoV/2) gives the half-extent of the viewport at unit depth, so
        // a 1.35m gate-radius spans gateHalfExtent / (distance * tan(VFoV/2))
        // of the viewport vertically. We use the same value for X — minor
        // approximation but the gate is square and Y is the tighter axis.
        float tanHalfV = Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
        float halfSizeV = gateHalfExtent / Mathf.Max(distance * tanHalfV, 0.001f);

        // Shrink the inset by allowedClipFraction so the user can opt in to a
        // little edge-case clipping if they want it.
        float inset = halfSizeV * (1f - allowedClipFraction);
        inset = Mathf.Clamp(inset, 0f, 0.49f);   // never collapse to a single point

        float u = Random.Range(inset, 1f - inset);
        float v = Random.Range(inset, 1f - inset);

        // ViewportToWorldPoint's z is distance along the camera's forward axis.
        gate.position = cam.ViewportToWorldPoint(new Vector3(u, v, distance));

        // Face the camera, with random tilt
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
    // Draws the near/far viewport rectangles gate centers can sample within.
    // The rectangles shrink with distance because closer gates need more inset
    // to keep their outer corners on-screen.
    void OnDrawGizmosSelected() {
        var cam = Camera.main;
        if (cam == null) return;

        Gizmos.color = Color.yellow;
        DrawViewportRect(cam, minDistance);
        DrawViewportRect(cam, maxDistance);
    }

    void DrawViewportRect(Camera cam, float distance) {
        float tanHalfV = Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
        float halfSizeV = gateHalfExtent / Mathf.Max(distance * tanHalfV, 0.001f);
        float inset = Mathf.Clamp(halfSizeV * (1f - allowedClipFraction), 0f, 0.49f);
        float lo = inset;
        float hi = 1f - inset;
        Vector3 tl = cam.ViewportToWorldPoint(new Vector3(lo, hi, distance));
        Vector3 tr = cam.ViewportToWorldPoint(new Vector3(hi, hi, distance));
        Vector3 br = cam.ViewportToWorldPoint(new Vector3(hi, lo, distance));
        Vector3 bl = cam.ViewportToWorldPoint(new Vector3(lo, lo, distance));
        Gizmos.DrawLine(tl, tr); Gizmos.DrawLine(tr, br);
        Gizmos.DrawLine(br, bl); Gizmos.DrawLine(bl, tl);
    }
}

[AttributeUsage(AttributeTargets.Method)]
public class InspectorButtonAttribute : Attribute {
    public string Label;
    public InspectorButtonAttribute(string label = null) { Label = label; }
}

#if UNITY_EDITOR
[CustomEditor(typeof(Randomizer))]
public class RandomizeGatePoseEditor : Editor {
    public override void OnInspectorGUI() {
        DrawDefaultInspector();
        if (GUILayout.Button("Randomize Now")) {
            ((Randomizer)target).Randomize();
        }
    }
}
#endif