using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using UnityEngine.Events;

public class ReceiveTCP : MonoBehaviour {
    [Header("Connection")]
    public int port = 9000;
    public bool listenOnStart = true;

    [Header("Events")]
    public UnityEvent<byte[]> OnMessageReceived;

    UdpClient _udp;
    Thread _thread;
    ConcurrentQueue<(int, byte[])> _queue = new ConcurrentQueue<(int, byte[])>();
    volatile bool _running;
    WorldModel _model;

    void Start() {
        _model = GetComponent<WorldModel>();
        if (listenOnStart) StartListening();
    }

    public void StartListening() {
        if (_running) return;
        _running = true;
        _udp = new UdpClient(port);
        _thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "UdpReceiver" };
        _thread.Start();
        Debug.Log($"[UdpReceiver] Listening on port {port}");
    }

    public void StopListening() {
        _running = false;
        _udp?.Close();
        _thread?.Join(500);
        Debug.Log("[UdpReceiver] Stopped.");
    }

    void Update() {
        while (_queue.TryDequeue(out var msg)) {
            var (topic, payload) = msg;
            if (topic == 1)
                _model?.OnMessage(payload);
            OnMessageReceived?.Invoke(payload);
        }
    }

    void OnDestroy() => StopListening();

    void ReceiveLoop() {
        var remote = new IPEndPoint(IPAddress.Any, 0);
        while (_running) {
            try {
                byte[] data = _udp.Receive(ref remote);
                int topic = data[0];
                byte[] payload = new byte[data.Length - 1];
                Buffer.BlockCopy(data, 1, payload, 0, payload.Length);
                // Debug.Log($"[UdpReceiver] {remote}: topic={topic} len={payload.Length}");
                _queue.Enqueue((topic, payload));
            } catch (Exception e) when (_running) {
                Debug.LogWarning($"[UdpReceiver] {e.Message}");
            }
        }
    }
}
