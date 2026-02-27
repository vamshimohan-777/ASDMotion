def route_video(duration, Tmin=2.0, video_path=None):
    """
    Decide routing based on file type only.

    Rule: only GIF files route to the image path; all other files go to video.
    """
    if video_path:
        ext = str(video_path).lower()
        if ext.endswith(".gif"):
            return "image"
        return "video"
    return "video"
