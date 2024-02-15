import os
import argparse


def save_learners(path, savedir, name):
    """Save coco learners to .user file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
    """
    users = []
    with open(os.path.join(path, "users.txt"), "r") as f:
        for line in f:
            users.append(line.strip())

    with open(os.path.join(savedir, name + ".user"), "w") as f:
        f.write("user_id:token\n")
        for user_id in users:
            f.write(f"{user_id}\n")
    return users


def save_courses(path, savedir, name):
    """Save coco courses to .item file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
    """

    courses = []
    with open(os.path.join(path, "courses.txt"), "r") as f:
        for line in f:
            courses.append(line.strip())

    with open(os.path.join(savedir, name + ".item"), "w") as f:
        f.write("item_id:token\n")
        for item_id in courses:
            f.write(f"{item_id}\n")
    return courses


def save_course_entity(path, savedir, name, courses):
    """Save coco course entity to .link file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        courses (list): list of courses
    """
    with open(os.path.join(savedir, name + ".link"), "w") as f:
        f.write("item_id:token\tentity_id:token\n")
        for course in courses:
            f.write(f"{course}\t{course}\n")


def read_concepts(path):
    """Save coco courses to .item file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
    """

    concepts = []
    with open(os.path.join(path, "concepts.txt"), "r") as f:
        for line in f:
            concepts.append(line.strip())

    return concepts


def read_schools(path):
    """Save coco courses to .item file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
    """

    schools = []
    with open(os.path.join(path, "schools.txt"), "r") as f:
        for line in f:
            schools.append(line.strip())

    return schools


def read_teachers(path):
    """Save coco courses to .item file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
    """

    teachers = []
    with open(os.path.join(path, "teachers.txt"), "r") as f:
        for line in f:
            teachers.append(line.strip())

    return teachers


def read_course_concepts(path, kg_triplets, courses, concepts):
    """Update kg triplets with instructors.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "course_concepts.txt"), "r") as f:
        for i, line in enumerate(f):
            course_concepts = line.strip()
            if course_concepts:
                for concept in course_concepts.split():
                    course = courses[int(i)]
                    kg_triplets.append(
                        [
                            course,
                            "has_concept",
                            concepts[int(concept)],
                        ]
                    )


def read_course_school(path, kg_triplets, courses, schools):
    """Update kg triplets with instructors.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "course_school.txt"), "r") as f:
        for i, line in enumerate(f):
            course_schools = line.strip()
            if course_schools:
                for school in course_schools.split():
                    course = courses[int(i)]
                    kg_triplets.append(
                        [
                            course,
                            "in_school",
                            schools[int(school)],
                        ]
                    )


def read_course_teachers(path, kg_triplets, courses, teachers):
    """Update kg triplets with instructors.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
    """
    with open(os.path.join(path, "course_teachers.txt"), "r") as f:
        for i, line in enumerate(f):
            course_teachers = line.strip()
            if course_teachers:
                for teacher in course_teachers.split():
                    course = courses[int(i)]
                    kg_triplets.append(
                        [
                            course,
                            "has_teacher",
                            teachers[int(teacher)],
                        ]
                    )


def save_kg_triplets(kg_triplets, path, name):
    """Save kg_triplets to file.

    Args:
        kg_triplets (list): list of triplets as a tuple (head_id, relation_id, tail_id)
        path (str): path to save the file
    """
    with open(os.path.join(path, name + ".kg"), "w") as f:
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")
        for head_id, relation_id, tail_id in kg_triplets:
            f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")


def save_enrolment(path, savedir, name, subset, learners, courses):
    """Save coco enrolments to file.

    Args:
        path (str): path of the dataset
        name (str): name of the dataset
        subset (str): name of the subset
        learners (list): list of learners
        courses (list): list of courses
    """
    enrolments = []
    with open(os.path.join(path, subset + ".txt"), "r") as f:
        for line in f:
            enrolments.append([int(x) for x in line.split()])

    with open(os.path.join(savedir, name + "." + subset + ".inter"), "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\n")
        for learner, course in enrolments:
            f.write(f"{learners[learner]}\t{courses[course]}\t1\n")


def format_pgpr_moocube(datadir, savedir, dataset_name):
    """Format PGPR-COCO dataset to recbole format.

    Args:
        datadir (str): path to the dataset
    """
    learners = save_learners(datadir, savedir, dataset_name)
    courses = save_courses(datadir, savedir, dataset_name)

    subsets = ["train", "validation", "test"]

    for subset in subsets:
        save_enrolment(datadir, savedir, dataset_name, subset, learners, courses)

    save_course_entity(datadir, savedir, dataset_name, courses)
    concepts = read_concepts(datadir)
    teachers = read_teachers(datadir)
    schools = read_schools(datadir)
    kg_triplets = []
    read_course_concepts(datadir, kg_triplets, courses, concepts)
    read_course_school(datadir, kg_triplets, courses, schools)
    read_course_teachers(datadir, kg_triplets, courses, teachers)
    save_kg_triplets(kg_triplets, savedir, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="data/mooc/processed_files")
    parser.add_argument("--savedir", type=str, default="data/mooc/recbolemoocube")
    parser.add_argument("--dataset_name", type=str, default="recbolemoocube")
    args = parser.parse_args()

    # Creates the folder savedir if it does not exist
    os.makedirs(args.savedir, exist_ok=True)

    format_pgpr_moocube(args.datadir, args.savedir, args.dataset_name)
