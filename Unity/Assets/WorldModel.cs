using System;
using System.Collections.Generic;
using UnityEngine;

/// Topic 1 payload: object_id (uint8) | px py pz (3xfloat32 LE) | qx qy qz qw (4xfloat32 LE) = 29 bytes
public class WorldModel : MonoBehaviour {
    [SerializeField] List<GameObject> gates;

    public void OnMessage(byte[] payload) {
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
}
