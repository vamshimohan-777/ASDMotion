# ASDMotion detection role: This module contributes to the end-to-end ASD/micro-event detection pipeline.
# Comments are added to clarify why the core logic matters for reliable detection outputs.

def route_video(duration, Tmin=2.0, video_path=None):
    """
    Decide routing based on file type only.

    Rule: only GIF files route to the image path; all other files go to video.
    """
    # Branch behavior based on the current runtime condition.
    if video_path:
        # Compute `ext` for the next processing step.
        ext = str(video_path).lower()
        # Branch behavior based on the current runtime condition.
        if ext.endswith(".gif"):
            # Return the result expected by the caller.
            return "image"
        # Return the result expected by the caller.
        return "video"
    # Return the result expected by the caller.
    return "video"

