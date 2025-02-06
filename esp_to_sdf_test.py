from esp_to_sdf import ESPtoSDF   


def main():
    esp_from_sdf = ESPtoSDF()
    esp_from_sdf.make_charge_sdf(sdf_file="/Users/localadmin/Documents/Documents - CMDADMIN’s MacBook Air/visualize_esp_public/Theophylline.sdf")

if __name__ == "__main__":
    main()