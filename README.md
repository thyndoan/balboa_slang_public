# balboa

UCSD CSE 167 codebase
https://cseweb.ucsd.edu/~tzli/cse167/

This is a revised version of balboa that builds on [SlangPy](https://github.com/shader-slang/slangpy).

# Setup

Clone the repository:

```bash
git clone https://github.com/BachiLi/balboa_slang_public.git
cd balboa_slang_public
```

We use [uv](https://docs.astral.sh/uv/) to manage Python packages and environments. Go to the linked website to install it if you haven't. Once installed, run

```bash
uv sync
```

Then try

```bash
uv run python main.py 1_1
```

You should see a window with white background.

# Platforms

Balboa depends on [SlangPy](https://github.com/shader-slang/slangpy) as a GPU programming framework, and should work on Windows, Linux, and Apple Silicon Macs.

**Intel Macs:** SlangPy does not currently publish macOS x86_64 wheels, so setup may require building SlangPy from source. If you are using an Intel Mac and `uv sync` fails because no compatible SlangPy package is available, you may need to build SlangPy from source. Contact the staff/instructor if you run into issues.

# HW2_8 custom scene

- Location:
  [`scenes/hw2/hw2_8/`](scenes/hw2/hw2_8)

# HW3_7 custom scene

- Location:
  [`scenes/hw3/hw3_7`](scenes/hw3/hw3_7)
- Please run `uv run python main.py 3_6 scenes/hw3/hw3_7/cat_scene.json` to see the animation!

# HW4_7 custom scene

- Location of the scene:
  [`scenes/hw4/hw4_7`](scenes/hw4/hw4_7)
- Please run `uv run python main.py 4_6 scenes/hw4/hw4_7/cat_rope.py` to see the animation!
- Video: `outputs/hw_4_7.mp4`
- Cat model was made by [Tekila](https://www.cgtrader.com/free-3d-models/animal/other/voxel-cats)

# Contributions for all homeworks

- We used AI to help us understand the concepts visually for part 1-6.
- We also asked it to help us with the debugging.
- We also used AI to help us with the custom scene

## details for HW3

- One of our problem was our code didn’t apply xform_pt and xform_vec to every ray, so some rays weren’t transformed into object space. After the AI explained how these functions work, we fixed our code to use them correctly.

- Another problem we ran into was some bugs in interpolated normal in triangle mesh, so the images were rendered wrong.

## details for HW4

- At first, we were confused in part 1 because the balls slowly stopped bouncing, even though our code never tells them to stop. Turned out that because we update the velocity before the position, each bounce loses a little height. The handout mentioned this, but we found it helpful to see it with step by step example from LLMs.

- Another one we ran into was in part 6. When we rendered the low poly rock scene, the cloth touched the rock, but then it flew away (after around 1s) instead of staying on the rock. LLMs showed us that our algorithm compared every close triangle with the floor and spheres, so sometimes a triangle that was not the closest one was used as the contact plane, and it pushed the cloth the wrong way. We ended up fixing it by finding the closest triangle first, and then, if two triangles are at the same distance, we keep the one that shows the point is more outside.
