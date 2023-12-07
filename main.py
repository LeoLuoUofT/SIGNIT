import subprocess
import tkinter as tk
from tkinter import ttk
from threading import Thread, Event
from ttkthemes import ThemedTk
import os


def run_script(
    script_path, input_directory, output_callback, terminate_event, duration
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
        spark_submit_cmd += [
            "--conf",
            "spark.driver.maxResultSize=0",
            script_path,
            str(duration),
        ]

    # Continue with the existing code for 'SIGNIT convert' subprocess

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


def create_gui(terminate_event, subprocesses):
    root = ThemedTk(theme="plastik")  # Change the theme to "plastik" for dark mode
    root.title("SPARK PROGRAM")
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
        # Terminate all subprocesses
        for process in subprocesses:
            process.terminate()

    root.protocol("WM_DELETE_WINDOW", on_close)  # Bind close event to on_close

    # Add Start and Kill buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, pady=10)

    kill_button = tk.Button(
        button_frame,
        text="Return to Menu",
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
    subprocesses = []

    while True:
        print("Removing previous byproducts.")
        bashCommand = "rm -r byproducts/stream_inputs/* byproducts/stream_outputs/* byproducts/intermediate/* byproducts/video_frames/*"
        os.system(bashCommand + " > nul 2>&1")
        user_choice = input(
            "Enter 1 for 'SIGNIT stream', 2 for 'SIGNIT convert', 3 for 'SIGNIT Video convert', or q to quit: "
        )
        if user_choice.lower() == "q":
            # Terminate all subprocesses before exiting
            for process in subprocesses:
                process.terminate()
            os.system(bashCommand + " > nul 2>&1")
            print("Shutting Down. Thank you for using SIGNIT.")
            break

        if user_choice == "1":
            script_path = "SIGNIT_stream.py"
            while True:
                try:
                    duration = int(input("Enter duration in seconds: "))
                    if duration <= 0:
                        print("Please enter a positive integer value.")
                        continue
                    break  # Exit the loop if a valid positive integer is entered
                except ValueError:
                    print("Invalid input. Please enter a positive integer value.")
                    duration = None

            input_directory = input("Enter the streaming URL: ")
            name = input("Enter the name of the stream: ")
        elif user_choice == "3":
            video_location = input("Enter the name of video: ")
            bashCommand = f"ffmpeg -i {video_location} -vf format=gray -r 1/1 byproducts/video_frames/%03d.jpg"
            os.system(bashCommand)
            input_directory = "byproducts/video_frames"
            script_path = "SIGNIT_convert.py"
            duration = None
            name = None
        elif user_choice == "2":
            script_path = "SIGNIT_convert.py"
            duration = None
            name = None
            input_directory = input("Enter the input directory: ")
        else:
            print("invalid input please try again")
            continue

        terminate_event = Event()  # Change to threading.Event

        # Run the script in a separate thread
        if user_choice == "1":
            process = subprocess.Popen(
                ["python", "stream_input.py", input_directory, name, str(duration)]
            )
            subprocesses.append(process)
            print("Running Stream for ", duration, "seconds.")

        # Errors with terminating process
        print(
            "WARNING! CLOSING THE WINDOW BEFORE THE PROCESS(ES) FINISHES WILL RESULT IN ERRORS."
        )

        root, text_widget = create_gui(terminate_event, subprocesses)

        def gui_updater(line):
            text_widget.after(0, update_gui, text_widget, line)

        script_thread = Thread(
            target=run_script,
            args=(
                script_path,
                input_directory,
                gui_updater,
                terminate_event,
                duration,
            ),
        )
        script_thread.start()

        try:
            # Main loop to keep the GUI running
            root.mainloop()

        except KeyboardInterrupt:
            pass

        finally:
            # Terminate the script thread when the main loop exits
            terminate_event.set()
            script_thread.join()


if __name__ == "__main__":
    main()
