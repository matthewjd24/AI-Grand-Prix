using System;
using System.Collections.Generic;
using UnityEngine;

/// Topic 1 — per-gate pose
/// Payload: object_id (uint8) | px py pz (3xfloat32 LE) | qx qy qz qw (4xfloat32 LE) = 29 bytes
///
/// Topic 2 — gate count
/// Payload: count (uint8)
///
/// Topic 3 — drone attitude (Unity-frame quaternion)
/// Payload: qx qy qz qw (4xfloat32 LE) = 16 bytes
public class WorldModel : MonoBehaviour {
    [SerializeField] List<GameObject> gates;
    [SerializeField] GameObject drone;

    public void OnPose(byte[] payload) {
        if (payload.Length < 29) return;

        int id = payload[0];
        if (id >= gates.Count || gates[id] == null) return;

        float px = BitConverter.ToSingle(payload, 1);
        float py = BitConverter.ToSingle(payload, 5);
        float pz = BitConverter.ToSingle(payload, 9);
        float qx = BitConverter.ToSingle(payload, 13);
        float qy = BitConverter.ToSingle(payload, 17);
        float qz = BitConverter.ToSingle(payload, 21);
        float qw = BitConverter.ToSingle(payload, 25);

        gates[id].transform.position = new Vector3(px, py, pz);
        gates[id].transform.rotation = new Quaternion(qx, qy, qz, qw);
    }

    public void OnGateCount(int count) {
        for (int i = 0; i < gates.Count; i++) {
            if (gates[i] != null)
                gates[i].SetActive(i < count);
        }
    }

    public void OnAttitude(byte[] payload) {
        if (payload.Length < 16 || drone == null) return;

        float qx = BitConverter.ToSingle(payload, 0);
        float qy = BitConverter.ToSingle(payload, 4);
        float qz = BitConverter.ToSingle(payload, 8);
        float qw = BitConverter.ToSingle(payload, 12);
        drone.transform.rotation = new Quaternion(qx, qy, qz, qw);
    }
}
