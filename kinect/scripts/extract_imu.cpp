#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <input.mkv> <output_imu.txt>\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];

    k4a_playback_t playback_handle = NULL;
    if (k4a_playback_open(input_path, &playback_handle) != K4A_RESULT_SUCCEEDED) {
        printf("Failed to open recording: %s\n", input_path);
        return 1;
    }

    FILE *f_out = fopen(output_path, "w");
    if (!f_out) {
        printf("Failed to open output file: %s\n", output_path);
        k4a_playback_close(playback_handle);
        return 1;
    }

    // Write header
    fprintf(f_out, "timestamp_usec,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n");

    k4a_stream_result_t result = K4A_STREAM_RESULT_SUCCEEDED;
    k4a_imu_sample_t imu_sample;
    int count = 0;

    printf("Extracting IMU data...\n");

    while (result == K4A_STREAM_RESULT_SUCCEEDED) {
        result = k4a_playback_get_next_imu_sample(playback_handle, &imu_sample);
        if (result == K4A_STREAM_RESULT_SUCCEEDED) {
            fprintf(f_out, "%lu,%f,%f,%f,%f,%f,%f\n",
                    imu_sample.acc_timestamp_usec,
                    imu_sample.acc_sample.xyz.x,
                    imu_sample.acc_sample.xyz.y,
                    imu_sample.acc_sample.xyz.z,
                    imu_sample.gyro_sample.xyz.x,
                    imu_sample.gyro_sample.xyz.y,
                    imu_sample.gyro_sample.xyz.z);
            count++;
        }
    }

    if (result == K4A_STREAM_RESULT_EOF) {
        printf("Reached end of file. Extracted %d IMU samples.\n", count);
    } else {
        printf("Failed to read IMU sample. Error code: %d\n", result);
    }

    fclose(f_out);
    k4a_playback_close(playback_handle);
    return 0;
}
