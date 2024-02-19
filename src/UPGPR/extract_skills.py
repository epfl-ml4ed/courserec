import os
import spacy
import argparse

import pandas as pd
from tqdm import tqdm
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor


def get_skills(skill_extractor, target_text):
    """Return the list of skills extracted from the target text

    Args:
        skill_extractor (SkillExtractor): skillNer object to extract the skills
        target_text (str): text to extract the skills from

    Returns:
        list: list of skills extracted from the target text
    """
    skills = []
    try:
        annotations = skill_extractor.annotate(target_text)
        skills += [
            skill["skill_id"] for skill in annotations["results"]["full_matches"]
        ]
        skills += [
            skill["skill_id"] for skill in annotations["results"]["ngram_scored"]
        ]
    except IndexError:
        pass
    except ValueError:
        pass

    return skills


def get_skills_from_course(datadir, skill_extractor):
    """Extract skills fron courses descriptions

    Args:
        datadir (str): path of the processed dataset
        skill_extractor (skillNer.skill_extractor_class.SkillExtractor): skill extractor

    Returns:
        pandas.dataframe: datadrame with courses and extracted skills
    """
    # Load the course_latest.csv file
    course_latest_df = pd.read_csv(os.path.join(datadir, "course_latest.csv"))

    # Remove all rows in which the short description is NaN
    course_skill_df = course_latest_df[course_latest_df["short_description"].notna()]

    # Remove all rows in which the language is not english
    course_skill_df = course_skill_df[course_skill_df.language == "english"]
    course_skill_df = course_skill_df[["course_id", "short_description"]]

    # for all rows, extract the skills from the short description
    course_skill_df["skills"] = course_skill_df.progress_apply(
        lambda row: get_skills(skill_extractor, row["short_description"]), axis=1
    )

    # Remove short descriptions
    course_skill_df.drop("short_description", axis=1, inplace=True)

    # Remake the dataframe so that each row contains a course and a skill
    course_skill_df = course_skill_df.explode("skills", ignore_index=True)

    # Remove all rows in which the skill is NaN
    course_skill_df = course_skill_df[course_skill_df.skills.notna()]

    return course_skill_df


def get_skills_from_learner(datadir, course_skill_df):
    """Associates learners with the skills they learned from the courses they have taken

    Args:
        datadir (str): path of the processed dataset
        course_skill_df (pandas.dataframe): dataframe of courses and their associated skills

    Returns:
        pandas.dataframe: dataframe of learners and their associated skills
    """
    enrolements_df = pd.read_csv(os.path.join(datadir, "evaluate_latest.csv"))
    enrolements_df = enrolements_df[["learner_id", "course_id"]]

    # Create the student_skills dataframe by merging the student_enrolments and course_skills_df dataframes
    learner_skills_df = pd.merge(
        enrolements_df, course_skill_df, on="course_id", how="left"
    )
    # Drop the course_id column
    learner_skills_df.drop("course_id", axis=1, inplace=True)
    # Drop the duplicates
    learner_skills_df.drop_duplicates(inplace=True)
    # remove rows where skills are null
    learner_skills_df.dropna(inplace=True)

    return learner_skills_df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str,
        default="data/coco/coco",
        help="Path to the processed dataset",
    )

    args = parser.parse_args()

    tqdm.pandas()

    # Load the spacy model and the skill extractor
    nlp = spacy.load("en_core_web_lg")
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

    # Get the course_skill dataframe
    course_skill_df = get_skills_from_course(args.datadir, skill_extractor)
    course_skill_df.to_csv(os.path.join(args.datadir, "course_skill.csv"))

    # Get the learner_skill dataframe
    learner_skills_df = get_skills_from_learner(args.datadir, course_skill_df)
    learner_skills_df.to_csv(os.path.join(args.datadir, "learner_skill.csv"))


if __name__ == "__main__":
    main()
