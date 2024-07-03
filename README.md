# TFM project
## Creating an image editor combining GroundingDino + SAM + Stable Diffusion
### Context 
In the era of Big Data, Data Science (DS), and Artificial Intelligence (AI), extracting knowledge and creativity has become a fundamental discipline, especially in image editing, which plays a crucial role in multiple areas like graphic design, advertising, marketing, and entertainment. However, manual editing can be tedious and time-consuming.

To address this problem, there is a need for automated tools that can facilitate and optimize the editing process. There have been incredible advances in AI, particularly in the field of generative models, presenting a unique opportunity to develop automatic image editors capable of performing complex tasks with greater efficiency and precision.

With the emergence of technologies such as Text-to-Image (TTI), segmentation models, and generative AI, different tools have shown remarkable performance in various tasks. Combining these technologies can pave the way for creating innovative tools that benefit everyone.

### Proposal
We propose to focus this project on developing an automated image editor that combines the power of three pre-trained AI models: GroundingDINO, SAM (Segment Anything Model), and Stable Diffusion. Each of these models contributes unique capabilities that, when combined, create a versatile and powerful tool for image editing. Here is a brief explanation of each model:

* GroundingDINO: This model excels in understanding and representing semantic concepts from images. It allows the automated editor to identify objects, scenes, and spatial relationships in images, which is fundamental for precise editing tasks.

* SAM (Segment Anything Model): This is a self-supervised attention model that enables the automated editor to focus on specific areas of an image in great detail. This is useful for tasks such as removing unwanted objects, editing fine details, or applying specific effects to certain parts of the image.

* Stable Diffusion: This is a state-of-the-art text-to-image diffusion model that allows the automated editor to generate new images from textual descriptions. This opens up a range of possibilities for creating personalized images and creative 
editing.


# To-Do List

## Project Tasks
- [X] Develop a jupyternotebook image editor
- [X] Integrate GroundingDINO model
- [X] Integrate SAM model
- [X] Integrate Stable Diffusion model
- [X] Combine the 3 in a single py file that is used with terminal
- [ ] Create a demoapp with FastAPI v1 (Fully functional no editing image implemented)
- [ ] Crate demo app v2, with the editing implementes
- [ ] Look for improvments

## Testing
- [ ] Perform unit tests
- [ ] Conduct user acceptance testing
- [ ] Evaluate performance with various test cases

## Documentation
- [ ] Write project proposal
- [x] Complete introduction section
- [x] Draft methodology section
- [ ] Models explanation
- [ ] Diragrams
- [ ] Finalize README file
