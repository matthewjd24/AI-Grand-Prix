using System.IO;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;

public class TransparentScreenshot : MonoBehaviour {

    [Header("Capture Settings")]
    public int width = 640;
    public int height = 360;
    public string outputFolder = "Renders";

    [InspectorButton("Capture Frame")]
    public void Capture() {
        var captureCamera = Camera.main;

        // Save the camera's original settings so we can restore them
        var originalClearFlags = captureCamera.clearFlags;
        var originalBackgroundColor = captureCamera.backgroundColor;
        var originalTargetTexture = captureCamera.targetTexture;

        // Configure camera for transparent capture
        captureCamera.clearFlags = CameraClearFlags.SolidColor;
        captureCamera.backgroundColor = new Color(0, 0, 0, 0);   // alpha 0 = transparent

        // Create a render texture with an alpha channel
        var rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        rt.antiAliasing = 8;
        captureCamera.targetTexture = rt;

        // Render into the texture
        captureCamera.Render();

        // Read the pixels back into a Texture2D
        RenderTexture.active = rt;
        var tex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();

        // Restore camera settings
        captureCamera.clearFlags = originalClearFlags;
        captureCamera.backgroundColor = originalBackgroundColor;
        captureCamera.targetTexture = originalTargetTexture;
        RenderTexture.active = null;
        rt.Release();
        Object.DestroyImmediate(rt);

        // Encode and save as PNG
        byte[] pngBytes = tex.EncodeToPNG();
        Object.DestroyImmediate(tex);

        string folder = Path.Combine(Application.dataPath, "..", outputFolder);
        Directory.CreateDirectory(folder);

        string filename = $"gate_{System.DateTime.Now:yyyyMMdd_HHmmss_fff}.png";
        string fullPath = Path.Combine(folder, filename);
        File.WriteAllBytes(fullPath, pngBytes);

        Debug.Log($"Saved: {fullPath}");
    }
}

#if UNITY_EDITOR
[CustomEditor(typeof(TransparentScreenshot))]
public class TransparentScreenshotButton : Editor {
    public override void OnInspectorGUI() {
        DrawDefaultInspector();

        if (GUILayout.Button("Capture")) {
            ((TransparentScreenshot)target).Capture();
        }
    }
}
#endif