import os
import pygame


class AlertSystem:
    def __init__(self, sound_path: str, volume: float = 0.8):
        self.sound_path = sound_path
        self.volume = volume
        self.initialized = False
        self.sound_loaded = False
        self.is_playing = False

        self._init_audio()

    def _init_audio(self):
        try:
            pygame.mixer.init()
            self.initialized = True

            if os.path.exists(self.sound_path):
                pygame.mixer.music.load(self.sound_path)
                pygame.mixer.music.set_volume(self.volume)
                self.sound_loaded = True
            else:
                print(f"[WARN] Alarm sound file not found: {self.sound_path}")
        except Exception as e:
            print(f"[WARN] Audio init failed: {e}")
            self.initialized = False
            self.sound_loaded = False

    def start_alert(self):
        if not self.initialized or not self.sound_loaded:
            return

        if not self.is_playing:
            try:
                pygame.mixer.music.play(-1)  # loop forever
                self.is_playing = True
            except Exception as e:
                print(f"[WARN] Failed to start alarm: {e}")

    def stop_alert(self):
        if not self.initialized or not self.sound_loaded:
            return

        if self.is_playing:
            try:
                pygame.mixer.music.stop()
                self.is_playing = False
            except Exception as e:
                print(f"[WARN] Failed to stop alarm: {e}")

    def cleanup(self):
        try:
            if self.initialized:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception:
            pass