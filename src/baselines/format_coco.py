import os
import argparse


def save_learners(datadir, savedir, name):
    """Save learners with recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of recbole dataset
        name (str): name of the dataset

    Returns:
        list: list of users
    """
    users = []
    with open(os.path.join(datadir, "learners.txt"), "r") as f:
        for line in f:
            users.append(line.strip())

    with open(os.path.join(savedir, name + ".user"), "w") as f:
        f.write("user_id:token\n")
        for user_id in users:
            f.write(f"{user_id}\n")
    return users


def save_courses(datadir, savedir, name):
    """Save courses with recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of recbole dataset
        name (str): name of the dataset

    Returns:
        list: list of courses
    """
    courses = []
    with open(os.path.join(datadir, "courses.txt"), "r") as f:
        for line in f:
            courses.append(line.strip())

    with open(os.path.join(savedir, name + ".item"), "w") as f:
        f.write("item_id:token\n")
        for item_id in courses:
            f.write(f"{item_id}\n")
    return courses


def save_course_entity(savedir, name, courses):
    """Save coco course entity to to recbole format.

    Args:
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
        courses (list): list of courses
    """
    with open(os.path.join(savedir, name + ".link"), "w") as f:
        f.write("item_id:token\tentity_id:token\n")
        for course in courses:
            f.write(f"{course}\tE_{course}\n")


def read_course_instructors(datadir, kg_triplets, courses):
    """Update kg triplets with instructors.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(datadir, "course_instructor.txt"), "r") as f:
        for i, line in enumerate(f):
            instructor = line.strip()
            if instructor:
                kg_triplets.append(
                    [
                        "E_" + courses[int(i)],
                        "instructor",
                        "Instructor_" + instructor,
                    ]
                )


def read_course_category(datadir, kg_triplets, courses):
    """Update kg triplets to with category.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(datadir, "course_scategory.txt"), "r") as f:
        for i, line in enumerate(f):
            category = line.strip()
            if category:
                kg_triplets.append(
                    [
                        "E_" + courses[int(i)],
                        "category",
                        "Category_" + category,
                    ]
                )


def read_course_skills(datadir, kg_triplets, courses):
    """Update kg triplets with skills.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(datadir, "course_skills.txt"), "r") as f:
        for i, line in enumerate(f):
            skills = line.strip()
            if skills:
                for skill in skills.split():
                    kg_triplets.append(
                        [
                            "E_" + courses[int(i)],
                            "skill",
                            "Skill_" + skill,
                        ]
                    )


def read_category_hierarchy(datadir, kg_triplets):
    """Update kg triplets to with category hierarchy.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
    """
    with open(os.path.join(datadir, "scategory_fcategory.txt"), "r") as f:
        for i, line in enumerate(f):
            pcategory = line.strip()
            if pcategory:
                kg_triplets.append(
                    [
                        "Category_" + str(i),
                        "child_category",
                        "ParentCategory_" + pcategory,
                    ]
                )


def save_kg_triplets(kg_triplets, savedir, name):
    """Save kg_triplets to file.

    Args:
        kg_triplets (list): list of triplets as a tuple (head_id, relation_id, tail_id)
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """
    with open(os.path.join(savedir, name + ".kg"), "w") as f:
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")
        for head_id, relation_id, tail_id in kg_triplets:
            f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")


def save_enrolment(datadir, savedir, name, subset, learners, courses):
    """Save coco enrolments to file.

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
        subset (str): name of the subset
        learners (list): list of learners
        courses (list): list of courses
    """
    enrolments = []
    with open(os.path.join(datadir, subset + ".txt"), "r") as f:
        for line in f:
            enrolments.append([int(x) for x in line.split()])

    with open(os.path.join(savedir, name + "." + subset + ".inter"), "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\n")
        for learner, course in enrolments:
            f.write(f"{learners[learner]}\t{courses[course]}\t1\n")


def format_pgpr_coco(datadir, savedir, dataset_name):
    """Format PGPR dataset to recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        dataset_name (str): name of the dataset
    """
    learners = save_learners(datadir, savedir, dataset_name)
    courses = save_courses(datadir, savedir, dataset_name)

    subsets = ["train", "validation", "test"]

    for subset in subsets:
        save_enrolment(datadir, savedir, dataset_name, subset, learners, courses)

    save_course_entity(savedir, dataset_name, courses)
    kg_triplets = []
    read_course_instructors(datadir, kg_triplets, courses)
    read_course_category(datadir, kg_triplets, courses)
    read_course_skills(datadir, kg_triplets, courses)
    read_category_hierarchy(datadir, kg_triplets)
    save_kg_triplets(kg_triplets, datadir, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="data/coco/processed_files")
    parser.add_argument("--savedir", type=str, default="data/coco/recbolecoco")
    parser.add_argument("--dataset_name", type=str, default="recbolecoco")

    args = parser.parse_args()

    # Creates the folder savedir if it does not exist
    os.makedirs(args.savedir, exist_ok=True)

    format_pgpr_coco(args.datadir, args.savedir, args.dataset_name)
