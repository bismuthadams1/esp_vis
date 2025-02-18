from esp_visualize_from_sdf import ESPFromSDF


def main():
    esp_from_sdf = ESPFromSDF()
    esp_from_sdf.process_and_launch_esp(sdf_file="Theophylline.sdf")


if __name__ == "__main__":
    main()