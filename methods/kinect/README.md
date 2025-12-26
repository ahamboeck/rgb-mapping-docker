# Azure Kinect Docker Container

This container provides tools for recording and processing data from Azure Kinect DK.

## 🔗 Official Resources
- **SDK**: [https://github.com/microsoft/Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)

## 🚀 Usage

### 1. Setup (Host)
You must install udev rules on your host machine to allow the container to access the USB device.
```bash
sudo cp 99-k4a.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 2. Build & Start
```bash
./run.sh build
./run.sh run
```

### 3. Example Commands
Inside the container:

**Viewer:**
```bash
k4aviewer
```

**Recording (Wide FOV):**
```bash
./scripts/record_wide.sh output_filename
```

**Recording (Narrow FOV):**
```bash
./scripts/record_narrow.sh output_filename
```

**Convert MKV to Image Sequence:**
```bash
python3 scripts/convert_kinect_k4a.py --input /workspace/output/recording.mkv --output /workspace/output/extracted_data
```
