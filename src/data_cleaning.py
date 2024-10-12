import os
import re


def clean_srt_to_txt(file_content):
    cleaned_lines = []
    skip_next_line = False
    for line in file_content.splitlines():
        if re.match(r"^\d+$", line.strip()):
            skip_next_line = True
            continue
        if skip_next_line:
            skip_next_line = False
            continue
        if line.strip():
            cleaned_lines.append(line)
    return " ".join(cleaned_lines)


# Get the current directory
# print(current_dir)
if __name__ == "__main__":
    current_dir = os.path.join(os.getcwd(), "out")
    # Loop through all files in the current directory
    for project in os.listdir(current_dir):
        print("project: ", project)
        project_path = os.path.join(current_dir, project, "compositions")
        if os.path.isdir(project_path):
            for composition in os.listdir(project_path):
                composition_path = os.path.join(project_path, composition)
                if os.path.isdir(composition_path):
                    for file in os.listdir(composition_path):
                        if file.endswith(".srt"):
                            srt_path = os.path.join(composition_path, file)
                            print("srt_path: ", srt_path)
                            print("composition: ", composition)

                            txt_path = os.path.join(
                                composition_path, file[:-4] + ".txt"
                            )

                            # Read the SRT file
                            with open(srt_path, "r", encoding="utf-8") as file:
                                srt_content = file.read()

                            # Clean the content
                            cleaned_txt_content = clean_srt_to_txt(srt_content)

                            # Write the cleaned content to a new TXT file
                            with open(txt_path, "w", encoding="utf-8") as file:
                                file.write(cleaned_txt_content)

                            print(f"Processed {srt_path} -> {txt_path}")

    print("All .srt files have been processed and converted to .txt files.")
