import subprocess
import tkinter as tk
from tkinter import ttk
from threading import Thread, Event
from ttkthemes import ThemedTk


def run_script(
    script_path, input_directory, output_callback, terminate_event, duration=None
):
    spark_submit_cmd = ["spark-submit", "--master", "local[*]"]

    if script_path == "SIGNIT_convert.py":
        spark_submit_cmd += [
            "--conf",
            "spark.driver.maxResultSize=0",
            script_path,
            input_directory,
        ]
        print("Running -> folder to Sign Language Letter conversion.")
    else:
        if duration is not None:
            spark_submit_cmd += [
                "--conf",
                "spark.driver.maxResultSize=0",
                script_path,
                input_directory,
                str(duration),
            ]
        else:
            spark_submit_cmd += [
                "--conf",
                "spark.driver.maxResultSize=0",
                script_path,
                input_directory,
            ]

    process = subprocess.Popen(
        spark_submit_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,  # Line buffered
    )

    # Read and process each line from the subprocess output
    for line in process.stdout:
        if terminate_event.is_set():
            break
        output_callback(line)

    process.wait()


def create_gui(terminate_event):
    root = ThemedTk(theme="plastik")  # Change the theme to "plastik" for dark mode
    root.title("Subprocess Output Viewer")
    root.geometry("600x600")

    text_frame = ttk.Frame(root)
    text_frame.pack(expand=True, fill="both")

    text_widget = tk.Text(
        text_frame,
        wrap="word",
        state="normal",
        background="#2E2E2E",
        foreground="white",
        insertbackground="white",
        font=("Courier", 12),
    )
    text_widget.pack(expand=True, fill="both")

    style = ttk.Style()
    style.configure("TButton", background="#3E3E3E", foreground="white")

    def on_close():
        terminate_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)  # Bind close event to on_close

    # Add Start and Kill buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, pady=10)

    kill_button = tk.Button(
        button_frame,
        text="Kill Subprocess",
        command=on_close,
        background="#3E3E3E",
        foreground="white",
    )
    kill_button.pack(padx=5)

    return root, text_widget


def update_gui(text_widget, line):
    text_widget.insert(tk.END, line)
    text_widget.see(tk.END)  # Scroll to the end

def main():
    while True:
        user_choice = input(
            "Enter 1 for 'SIGNIT stream', 2 for 'SIGNIT convert', or q to quit: "
        )
        if user_choice.lower() == "q":
            break

        if user_choice == "1":
            script_path = "SIGNIT_stream.py"
            try:
                duration = int(
                    input("Enter duration in seconds (or press Enter for unlimited): ")
                )
            except ValueError:
                duration = None  # If the user enters an invalid value, run without duration limit

        else:
            script_path = "SIGNIT_convert.py"
            duration = None

        input_directory = input("Enter the input directory: ")

        terminate_event = Event()  # Change to threading.Event

        root, text_widget = create_gui(terminate_event)

        def gui_updater(line):
            text_widget.after(0, update_gui, text_widget, line)

        # Run the script in a separate thread
        script_thread = Thread(
            target=run_script,
            args=(script_path, input_directory, gui_updater, terminate_event, duration),
        )
        script_thread.start()

        return_to_menu = False

        def check_input():
            nonlocal return_to_menu
            if input("Do you want to return to the main menu? (y/n): ").lower() == "y":
                return_to_menu = True
                root.quit()

        # Check for user input periodically
        root.after(1000, check_input)

        try:
            # Main loop to keep the GUI running
            root.mainloop()

            if return_to_menu:
                break

        except Exception:
            pass

        finally:
            # Terminate the script thread when the main loop exits
            terminate_event.set()
            script_thread.join()

        print("okay")



if __name__ == "__main__":
    main()
