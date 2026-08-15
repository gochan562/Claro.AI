from .notebook_builder import build_cells

def generate_notebook(model_id: str, task: str):
    loader, inference = build_cells(model_id, task)
    return {
        "cells": [loader, inference],
        "loader": loader,
        "inference": inference,
        "model_id": model_id,
        "task": task,
    }